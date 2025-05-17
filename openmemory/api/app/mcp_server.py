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
from typing import Optional

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

@mcp.tool(description="Add a new memory. If text is large or original_filename is provided, it might be stored as a file pointer.")
async def add_memories(text: str, original_filename: Optional[str] = None) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"}, indent=2)
    if not client_name:
        return json.dumps({"error": "client_name not provided"}, indent=2)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            if not user or not app:
                return json.dumps({"error": "User or App context not found"}, indent=2)
            if not app.is_active:
                return json.dumps({"error": f"App {app.name} is currently paused."}, indent=2)

            memory_text_to_store = text
            metadata_to_store = {
                "source_app": "openmemory",
                "mcp_client": client_name,
            }
            is_file_pointer = False

            if original_filename or len(text) > FILE_CONTENT_THRESHOLD:
                is_file_pointer = True
                user_file_dir = FILE_STORAGE_BASE_PATH / uid
                user_file_dir.mkdir(parents=True, exist_ok=True)
                
                actual_original_filename = original_filename or "large_text.txt"
                # Sanitize original_filename for use in stored_filename to avoid issues
                sane_original_filename = Path(actual_original_filename).name # Basic sanitization
                stored_filename = f"{uuid.uuid4()}_{sane_original_filename}"
                file_path = user_file_dir / stored_filename

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                file_pointer_data = {
                    "type": FILE_POINTER_MEMORY_TYPE,
                    "original_filename": actual_original_filename,
                    "stored_filename": stored_filename,
                    "file_path_in_container": str(file_path),
                    "size_bytes": len(text.encode('utf-8')),
                    "char_length": len(text)
                }
                memory_text_to_store = f"Stored file: {actual_original_filename} (Content pointer)"
                metadata_to_store.update(file_pointer_data) # Add file pointer info to metadata
                logging.info(f"Stored large text/file for user {uid} at {file_path}")

            # Call mem0 client to add the memory (either pointer or actual short text)
            response = memory_client.add(
                memory_text_to_store, 
                user_id=uid, 
                metadata=metadata_to_store
            )

            # Process the response and update database (mostly for the pointer memory or short memory)
            if isinstance(response, dict) and 'results' in response:
                for result in response['results']:
                    if not result.get('id') or not result.get('memory'): # Basic validation of mem0 result
                        logging.warning(f"Skipping incomplete result from mem0.add: {result}")
                        continue
                    
                    memory_id = uuid.UUID(result['id'])
                    sql_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    
                    determined_old_state_for_history = MemoryState.deleted # Default for NOT NULL constraint

                    if result['event'] == 'ADD':
                        if not sql_memory: # Truly new memory
                            determined_old_state_for_history = MemoryState.deleted # Satisfy NOT NULL if it means "wasn't active"
                            sql_memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                metadata_=metadata_to_store,
                                state=MemoryState.active
                            )
                            db.add(sql_memory)
                        else: # Memory existed in SQL, mem0 is 'ADD'ing it (treat as re-activation/update)
                            determined_old_state_for_history = sql_memory.state # Capture its previous state
                            sql_memory.state = MemoryState.active
                            sql_memory.content = result['memory']
                            sql_memory.metadata_ = metadata_to_store 
                            sql_memory.updated_at = datetime.datetime.now(datetime.UTC)

                        history = MemoryStatusHistory(
                            memory_id=memory_id, 
                            changed_by=user.id,
                            old_state=determined_old_state_for_history, 
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'UPDATE':
                        if sql_memory:
                            determined_old_state_for_history = sql_memory.state
                            sql_memory.content = result['memory']
                            sql_memory.metadata_ = metadata_to_store 
                            sql_memory.state = MemoryState.active 
                            sql_memory.updated_at = datetime.datetime.now(datetime.UTC)
                            history = MemoryStatusHistory(
                                memory_id=memory_id, 
                                changed_by=user.id,
                                old_state=determined_old_state_for_history, 
                                new_state=MemoryState.active
                            )
                            db.add(history)
                        else:
                            # This case (UPDATE for non-existent sql_memory) should ideally not happen
                            # If it does, treat as ADD for history purposes
                            determined_old_state_for_history = MemoryState.deleted # Default for NOT NULL
                            # Create the memory as it was supposed to be updated
                            sql_memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                metadata_=metadata_to_store,
                                state=MemoryState.active # Set to active as it is an update
                            )
                            db.add(sql_memory)
                            history = MemoryStatusHistory(
                                memory_id=memory_id, 
                                changed_by=user.id,
                                old_state=determined_old_state_for_history, 
                                new_state=MemoryState.active
                            )
                            db.add(history)
                db.commit()
            
            # Return the response from memory_client.add, which might contain pointer info
            return json.dumps(response if response else {"message": "Operation processed, no specific results from mem0 client."}, indent=2) 
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in add_memories: {e}")
        # Attempt to rollback in case of partial DB changes before error, though SessionLocal might handle this
        try:
            db.rollback()
        except Exception as rb_e:
            logging.error(f"Error during rollback: {rb_e}")
        return json.dumps({"error": f"Error adding memory: {str(e)}"}, indent=2)


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get accessible memory IDs based on ACL
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            
            conditions = [qdrant_models.FieldCondition(key="user_id", match=qdrant_models.MatchValue(value=uid))]
            
            if accessible_memory_ids:
                # Convert UUIDs to strings for Qdrant
                accessible_memory_ids_str = [str(memory_id) for memory_id in accessible_memory_ids]
                conditions.append(qdrant_models.HasIdCondition(has_id=accessible_memory_ids_str))

            filters = qdrant_models.Filter(must=conditions)
            embeddings = memory_client.embedding_model.embed(query, "search")
            
            hits = memory_client.vector_store.client.query_points(
                collection_name=memory_client.vector_store.collection_name,
                query=embeddings,
                query_filter=filters,
                limit=10,
            )

            # Process search results
            memories = hits.points
            memories = [
                {
                    "id": memory.id,
                    "memory": memory.payload["data"],
                    "hash": memory.payload.get("hash"),
                    "created_at": memory.payload.get("created_at"),
                    "updated_at": memory.payload.get("updated_at"),
                    "score": memory.score,
                }
                for memory in memories
            ]

            # Log memory access for each memory found
            if isinstance(memories, dict) and 'results' in memories:
                print(f"Memories: {memories}")
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="search",
                            metadata_={
                                "query": query,
                                "score": memory_data.get('score'),
                                "hash": memory_data.get('hash')
                            }
                        )
                        db.add(access_log)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    # Create access log entry
                    access_log = MemoryAccessLog(
                        memory_id=memory_id,
                        app_id=app.id,
                        access_type="search",
                        metadata_={
                            "query": query,
                            "score": memory.get('score'),
                            "hash": memory.get('hash')
                        }
                    )
                    db.add(access_log)
                db.commit()
            return json.dumps(memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(e)
        return f"Error searching memory: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get all memories
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            # Filter memories based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={
                                    "hash": memory_data.get('hash')
                                }
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={
                                "hash": memory.get('hash')
                            }
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        return f"Error getting memories: {e}"


@mcp.tool(description="Fetches the most recently added memory for the user. If it's a file pointer, attempts to return file content.")
async def get_last_memory() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return json.dumps({"error": "user_id not provided"}, indent=2)
    if not client_name:
        return json.dumps({"error": "client_name not provided"}, indent=2)

    try:
        db = SessionLocal()
        try:
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)
            if not user or not app:
                return json.dumps({"error": "User or App context not found"}, indent=2)

            all_memories_response = memory_client.get_all(user_id=uid)
            memories_list = []
            if isinstance(all_memories_response, dict) and 'results' in all_memories_response:
                memories_list = all_memories_response['results']
            elif isinstance(all_memories_response, list):
                memories_list = all_memories_response

            if not memories_list:
                return json.dumps({"message": "No memories found"}, indent=2)

            memories_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)

            for mem_data_from_mem0 in memories_list:
                if 'id' not in mem_data_from_mem0:
                    continue
                try:
                    memory_id_uuid = uuid.UUID(mem_data_from_mem0['id'])
                except ValueError:
                    logging.warning(f"Invalid UUID for memory ID {mem_data_from_mem0['id']}")
                    continue
                
                sql_memory_obj = db.query(Memory).filter(Memory.id == memory_id_uuid, Memory.user_id == user.id).first()

                if sql_memory_obj and check_memory_access_permissions(db, sql_memory_obj, app.id):
                    # Log access for the memory pointer itself
                    access_log = MemoryAccessLog(
                        memory_id=memory_id_uuid, app_id=app.id, access_type="get_last_pointer_check",
                        metadata_={'hash': mem_data_from_mem0.get('hash')}
                    )
                    db.add(access_log)
                    # db.commit() # Commit logs later or after successful file read

                    # Check if this memory is a file pointer using its SQL metadata
                    # The metadata from mem0.get_all() might not be complete or structured as we saved it in SQL.
                    # So, rely on sql_memory_obj.metadata_ which we stored during add_memories.
                    pointer_details = sql_memory_obj.metadata_ or {}
                    if pointer_details.get("type") == FILE_POINTER_MEMORY_TYPE and 'file_path_in_container' in pointer_details:
                        file_path_str = pointer_details['file_path_in_container']
                        original_filename = pointer_details.get('original_filename', Path(file_path_str).name)
                        try:
                            actual_file_path = Path(file_path_str)
                            if actual_file_path.is_file():
                                with open(actual_file_path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                logging.info(f"Returning content of file {actual_file_path} for memory {memory_id_uuid}")
                                db.commit() # Commit access log for pointer and successful file read
                                return json.dumps({
                                    "type": "file_content",
                                    "original_filename": original_filename,
                                    "stored_filename": pointer_details.get('stored_filename'),
                                    "content": file_content,
                                    "retrieved_at": datetime.datetime.now(datetime.UTC).isoformat(),
                                    "original_memory_id": str(memory_id_uuid),
                                    "pointer_details": pointer_details
                                }, indent=2)
                            else:
                                logging.error(f"File not found at path {actual_file_path} for memory {memory_id_uuid}")
                                # Fall through to return the pointer if file is missing
                        except Exception as e:
                            logging.exception(f"Error reading file {file_path_str} for memory {memory_id_uuid}: {e}")
                            # Fall through to return the pointer if file read fails
                    
                    # If not a file pointer or file operation failed/file not found, return the memory as is (pointer or regular memory)
                    db.commit() # Commit access log for the memory itself
                    return json.dumps(mem_data_from_mem0, indent=2) # Return the mem0 data which has the pointer text
            
            db.commit() # Commit if no accessible memories were found after looping
            return json.dumps({"message": "No accessible recent memories found"}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error in get_last_memory: {e}")
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}, indent=2)


@mcp.tool(description="Delete all memories in the user's memory. If a memory is a file pointer, only the pointer is deleted, not the underlying file.")
async def delete_all_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                memory_client.delete(memory_id)

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()
            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        return f"Error deleting memories: {e}"


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
