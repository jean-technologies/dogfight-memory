import logging
import json
import os
import uuid
import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from app.utils.memory import get_memory_client
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
import contextvars
from dotenv import load_dotenv
from app.database import SessionLocal
from app.models import Memory, MemoryState, MemoryStatusHistory, MemoryAccessLog
from app.utils.db import get_user_and_app
from app.utils.permissions import check_memory_access_permissions
from qdrant_client import models as qdrant_models
from typing import Optional, List

# Load environment variables
load_dotenv()

# Initialize MCP and memory client
mcp = FastMCP("mem0-mcp-server")

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise Exception("OPENAI_API_KEY is not set in .env file")

memory_client = get_memory_client()

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Initialize SSE transport
sse = SseServerTransport("/mcp/messages/")

# Constants for file handling
FILE_STORAGE_BASE_PATH = Path("/usr/src/openmemory/user_files") 
FILE_CONTENT_THRESHOLD = 4000  # Characters
FILE_POINTER_MEMORY_TYPE = "file_pointer_v1"

@mcp.tool(description="Add a new memory. If text is large or original_filename is provided, it is stored as a file pointer.")
async def add_memories(text: str, original_filename: Optional[str] = None) -> str:
    logging.info(f"[add_memories] Called with text length: {len(text)}, filename: {original_filename}")
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid: 
        logging.error("[add_memories] Error: user_id not provided")
        return json.dumps({"error": "user_id not provided"}, indent=2)
    if not client_name: 
        logging.error("[add_memories] Error: client_name not provided")
        return json.dumps({"error": "client_name not provided"}, indent=2)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            if not user or not app:
                logging.error(f"[add_memories] User or App not found for uid: {uid}, client: {client_name}")
                return json.dumps({"error": "User or App context not found"}, indent=2)
            if not app.is_active:
                logging.warning(f"[add_memories] App {app.name} is currently paused for user {uid}.")
                return json.dumps({"error": f"App {app.name} is currently paused."}, indent=2)

            memory_text_to_ingest_by_mem0 = text
            metadata_for_sql_and_mem0 = {
                "source_app": "openmemory",
                "mcp_client": client_name,
                "type": "direct_text" # Default type
            }

            if original_filename or len(text) > FILE_CONTENT_THRESHOLD:
                logging.info(f"[add_memories] Large text or filename provided. Storing as file pointer.")
                user_file_dir = FILE_STORAGE_BASE_PATH / uid
                user_file_dir.mkdir(parents=True, exist_ok=True)
                
                actual_original_filename = original_filename or "large_text_content.md"
                sane_original_filename = Path(actual_original_filename).name 
                stored_filename_uuid_part = str(uuid.uuid4())
                stored_filename = f"{stored_filename_uuid_part}_{sane_original_filename}"
                file_path_obj = user_file_dir / stored_filename

                with open(file_path_obj, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"[add_memories] File saved to: {file_path_obj}")
                
                # This is the text that mem0 will process for the pointer memory
                memory_text_to_ingest_by_mem0 = f"File stored: {actual_original_filename}. Original char length: {len(text)}."
                
                # This metadata will be associated with the pointer memory in mem0 and our SQL DB
                metadata_for_sql_and_mem0.update({
                    "type": FILE_POINTER_MEMORY_TYPE,
                    "original_filename": actual_original_filename,
                    "stored_filename": stored_filename, # The name of file in user_files_storage
                    "file_path_in_container": str(file_path_obj),
                    "size_bytes": len(text.encode('utf-8')),
                    "char_length": len(text)
                })
            else:
                logging.info(f"[add_memories] Storing as direct text memory.")

            logging.info(f"[add_memories] Calling memory_client.add for user {uid} with pointer/text: \"{memory_text_to_ingest_by_mem0[:100]}...\"")
            response_from_mem0 = memory_client.add(
                memory_text_to_ingest_by_mem0, 
                user_id=uid, 
                metadata=metadata_for_sql_and_mem0 # Pass the rich metadata here
            )
            logging.info(f"[add_memories] memory_client.add response: {response_from_mem0}")

            if isinstance(response_from_mem0, dict) and 'results' in response_from_mem0:
                for result_from_mem0 in response_from_mem0['results']:
                    if not result_from_mem0.get('id') or not result_from_mem0.get('memory'):
                        logging.warning(f"[add_memories] Skipping incomplete result from mem0.add: {result_from_mem0}")
                        continue
                    
                    memory_id = uuid.UUID(result_from_mem0['id'])
                    # The content from mem0.add result (result_from_mem0['memory']) is what mem0 processed/stored (e.g. the pointer text)
                    # The metadata_for_sql_and_mem0 is what WE want to ensure is in our SQL DB.

                    sql_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    determined_old_state_for_history = MemoryState.deleted
                    
                    if result_from_mem0['event'] == 'ADD':
                        if not sql_memory: # Truly new
                            # determined_old_state_for_history is already MemoryState.deleted
                            sql_memory = Memory(
                                id=memory_id, user_id=user.id, app_id=app.id,
                                content=result_from_mem0['memory'], # Content from mem0 (pointer text or direct short text)
                                metadata_=metadata_for_sql_and_mem0, # Our full metadata
                                state=MemoryState.active
                            )
                            db.add(sql_memory)
                        else: # Existed in SQL, mem0 says ADD (treat as update/reactivate)
                            determined_old_state_for_history = sql_memory.state
                            sql_memory.content = result_from_mem0['memory']
                            sql_memory.metadata_ = metadata_for_sql_and_mem0
                            sql_memory.state = MemoryState.active
                            sql_memory.updated_at = datetime.datetime.now(datetime.UTC)
                    
                    elif result_from_mem0['event'] == 'UPDATE':
                        if sql_memory:
                            determined_old_state_for_history = sql_memory.state
                            sql_memory.content = result_from_mem0['memory']
                            sql_memory.metadata_ = metadata_for_sql_and_mem0
                            sql_memory.state = MemoryState.active
                            sql_memory.updated_at = datetime.datetime.now(datetime.UTC)
                        else: # UPDATE for a memory not in SQL? Log warning, create it.
                            logging.warning(f"[add_memories] mem0 reported UPDATE for non-existent SQL memory ID {memory_id}. Creating.")
                            # determined_old_state_for_history is already MemoryState.deleted
                            sql_memory = Memory(id=memory_id, user_id=user.id, app_id=app.id, content=result_from_mem0['memory'], metadata_=metadata_for_sql_and_mem0, state=MemoryState.active)
                            db.add(sql_memory)
                    else:
                        logging.warning(f"[add_memories] Unhandled event type from mem0: {result_from_mem0.get('event')}")
                        continue # Skip history for unhandled events

                    history = MemoryStatusHistory(
                        memory_id=memory_id, changed_by=user.id,
                        old_state=determined_old_state_for_history, new_state=MemoryState.active
                    )
                    db.add(history)
                db.commit()
            else:
                logging.warning(f"[add_memories] mem0.add response did not contain 'results': {response_from_mem0}")

            return json.dumps(response_from_mem0 if response_from_mem0 else {"message": "Operation processed"}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"[add_memories] Top-level exception: {e}")
        try: db.rollback()
        except: pass
        return json.dumps({"error": f"Critical error in add_memories: {str(e)}"}, indent=2)

@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str) -> str:
    logging.info(f"[search_memory] Called with query: {query}")
    uid = user_id_var.get(None); client_name = client_name_var.get(None)
    if not uid or not client_name: return json.dumps({"error": "context missing"})
    # For now, keeping search simplified as it seemed to work for basic cases.
    # Restoring full search logic with DB permission checks would be the next step after add/get_last are stable.
    try: 
        logging.info("[search_memory] Calling memory_client.search...")
        response = memory_client.search(user_id=uid, query=query)
        logging.info(f"[search_memory] memory_client.search response: {response}")
        return json.dumps(response, indent=2)
    except Exception as e:
        logging.exception(f"[search_memory] Exception: {e}")
        return json.dumps({"error": f"Search error: {str(e)}"}) 

@mcp.tool(description="List all memories in the user\'s memory. (Simplified for debugging)")
async def list_memories() -> str:
    logging.info("[list_memories] Called (simplified version)")
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid: return json.dumps({"error": "user_id not provided"}, indent=2)
    if not client_name: return json.dumps({"error": "client_name not provided"}, indent=2)
    try:
        logging.info(f"[list_memories] Attempting to call memory_client.get_all for user {uid}")
        raw_response_from_mem0 = memory_client.get_all(user_id=uid)
        logging.info(f"[list_memories] memory_client.get_all response: {raw_response_from_mem0}")
        memories_to_return = []
        if isinstance(raw_response_from_mem0, dict) and 'results' in raw_response_from_mem0:
            memories_to_return = raw_response_from_mem0['results']
        elif isinstance(raw_response_from_mem0, list):
            memories_to_return = raw_response_from_mem0
        return json.dumps(memories_to_return, indent=2)
    except Exception as e:
        logging.exception(f"[list_memories] Exception: {e}")
        return json.dumps({"error": f"Error listing memories: {str(e)}"}, indent=2)

@mcp.tool(description="Fetches the most recently added memory. If it is a file pointer, returns file content.")
async def get_last_memory() -> str:
    logging.info("[get_last_memory] Called (full file-aware version)")
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid: return json.dumps({"error": "user_id not provided"}, indent=2)
    if not client_name: return json.dumps({"error": "client_name not provided"}, indent=2)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            if not user or not app: 
                logging.error(f"[get_last_memory] User or App not found for uid: {uid}, client: {client_name}")
                return json.dumps({"error": "User or App context not found"}, indent=2)

            logging.info(f"[get_last_memory] Calling memory_client.get_all for user {uid}")
            all_memories_data_from_mem0 = memory_client.get_all(user_id=uid)
            logging.info(f"[get_last_memory] memory_client.get_all raw response: {all_memories_data_from_mem0}")
            
            memories_list_from_mem0 = []
            if isinstance(all_memories_data_from_mem0, dict) and 'results' in all_memories_data_from_mem0:
                memories_list_from_mem0 = all_memories_data_from_mem0['results']
            elif isinstance(all_memories_data_from_mem0, list):
                memories_list_from_mem0 = all_memories_data_from_mem0
            # else: log or handle unexpected format if necessary

            if not memories_list_from_mem0:
                logging.info("[get_last_memory] No memories found from memory_client.get_all.")
                return json.dumps({"message": "No memories found"}, indent=2)

            # Sort by created_at from mem0 data to find the latest candidate
            memories_list_from_mem0.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            logging.info(f"[get_last_memory] Sorted {len(memories_list_from_mem0)} candidates from mem0.")

            for mem_candidate_from_mem0 in memories_list_from_mem0:
                if 'id' not in mem_candidate_from_mem0:
                    logging.warning(f"[get_last_memory] mem0 candidate missing id: {mem_candidate_from_mem0}")
                    continue
                try:
                    memory_id_uuid = uuid.UUID(mem_candidate_from_mem0['id'])
                except ValueError:
                    logging.warning(f"[get_last_memory] Invalid UUID for mem0 candidate ID: {mem_candidate_from_mem0['id']}")
                    continue
                
                # Fetch corresponding SQL Memory object to get our authoritative metadata and check permissions
                sql_memory_obj = db.query(Memory).filter(Memory.id == memory_id_uuid, Memory.user_id == user.id).first()

                if sql_memory_obj and check_memory_access_permissions(db, sql_memory_obj, app.id):
                    logging.info(f"[get_last_memory] Found accessible SQL memory: {sql_memory_obj.id}, type in metadata: {sql_memory_obj.metadata_.get('type') if sql_memory_obj.metadata_ else 'N/A'}")
                    # Log access for the memory pointer/record itself
                    access_log = MemoryAccessLog(
                        memory_id=memory_id_uuid, app_id=app.id, access_type="get_last_attempt",
                        metadata_={'hash': mem_candidate_from_mem0.get('hash')}
                    )
                    db.add(access_log)
                    # db.commit() # Commit later

                    authoritative_metadata = sql_memory_obj.metadata_ or {}
                    if authoritative_metadata.get("type") == FILE_POINTER_MEMORY_TYPE and 'file_path_in_container' in authoritative_metadata:
                        file_path_str = authoritative_metadata['file_path_in_container']
                        original_filename_from_meta = authoritative_metadata.get('original_filename', Path(file_path_str).name)
                        logging.info(f"[get_last_memory] Is file pointer. Path: {file_path_str}")
                        try:
                            actual_file_path = Path(file_path_str)
                            if actual_file_path.is_file():
                                with open(actual_file_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                logging.info(f"[get_last_memory] Successfully read file content from {actual_file_path}")
                                db.commit() # Commit access log for successful file read
                                return json.dumps({
                                    "type": "file_content",
                                    "original_filename": original_filename_from_meta,
                                    "stored_filename": authoritative_metadata.get('stored_filename'),
                                    "char_length": authoritative_metadata.get('char_length'),
                                    "content": file_content,
                                    "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
                                    "original_memory_id": str(memory_id_uuid),
                                    "pointer_details": authoritative_metadata # Include all pointer details
                                }, indent=2)
                            else:
                                logging.error(f"[get_last_memory] File not found at path {actual_file_path} (from metadata of memory {memory_id_uuid})")
                                # Fall through to return the pointer memory itself if file is missing
                        except Exception as e:
                            logging.exception(f"[get_last_memory] Error reading file {file_path_str} for memory {memory_id_uuid}: {e}")
                            # Fall through to return pointer memory if file read fails
                    
                    # If not a file pointer, or file operation failed, return the (pointer) memory as processed by mem0
                    logging.info(f"[get_last_memory] Returning memory (not file content): {mem_candidate_from_mem0['id']}")
                    db.commit() # Commit access log for the memory itself
                    return json.dumps(mem_candidate_from_mem0, indent=2)
            
            # If loop finishes, no accessible recent memory was found
            logging.info("[get_last_memory] No accessible recent memories found after checking all candidates.")
            db.commit() # Commit any pending logs from non-accessible checks if any
            return json.dumps({"message": "No accessible recent memories found"}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"[get_last_memory] Top-level exception: {e}")
        return json.dumps({"error": f"Critical error in get_last_memory: {str(e)}"}, indent=2)

@mcp.tool(description="Delete all memories in the user\'s memory. If a memory is a file pointer, only the pointer is deleted, not the underlying file.")
async def delete_all_memories() -> str:
    logging.info("[delete_all_memories] Called")
    uid = user_id_var.get(None); client_name = client_name_var.get(None)
    if not uid or not client_name: return json.dumps({"error": "context missing"})
    try:
        logging.info("[delete_all_memories] Placeholder for actual deletion logic.")
        return json.dumps({"message": "delete_all_memories (placeholder) called"})
    except Exception as e:
        logging.exception(f"[delete_all_memories] Exception: {e}")
        return json.dumps({"error": f"Delete error: {str(e)}"}) 

@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client"""
    # Extract user_id and client_name from path parameters
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")

    try:
        # Handle SSE connection
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)

@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)

@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_post_message(request: Request):
    return await handle_post_message(request)

async def handle_post_message(request: Request):
    """Handle POST messages for SSE"""
    try:
        body = await request.body()

        # Create a simple receive function that returns the body
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Create a simple send function that does nothing
        async def send(message):
            return {}

        # Call handle_post_message with the correct arguments
        await sse.handle_post_message(request.scope, receive, send)

        # Return a success response
        return {"status": "ok"}
    finally:
        pass
        # Clean up context variable
        # client_name_var.reset(client_token)

def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = f"mem0-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)
