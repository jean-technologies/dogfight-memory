# OpenMemory - Development Journal: Large File Storage & Retrieval

This document outlines the development process, challenges, and solutions implemented for handling large text/file storage and retrieval within the OpenMemory system, particularly focusing on the `add_memories` and `get_last_memory` MCP tools.

## Initial Goal

The primary objective was to enable users (e.g., an AI assistant like Claude interacting with the OpenMemory MCP server) to save large pieces of text (essays, documents) and reliably retrieve the latest one, primarily for a workflow where content is saved from one client (Claude) and retrieved by another (Cursor).

## Core Components Involved

-   **`add_memories` MCP Tool:** Responsible for receiving text (and an optional original filename) and storing it.
-   **`get_last_memory` MCP Tool:** Responsible for retrieving the most recently added memory/file.
-   **`mem0` Library (`memory_client`):** The underlying library used for creating memory records, which involves processing text, generating embeddings, and storing in a vector database (Qdrant) and an SQL database (SQLite for metadata and history).
-   **Physical File Storage:** A designated directory (`user_files_storage/user/<user_id>/`) for storing the actual content of large files.
-   **SQL Database:** Stores metadata about memories, including pointers to physical files.

## Iteration 1: Basic File Pointer Strategy

1.  **Concept:**
    *   If `add_memories` receives large text or an `original_filename`, save the content to a physical file.
    *   Pass a descriptive pointer string (e.g., "File stored: filename.md, Length: N") to `memory_client.add()`.
    *   Store detailed metadata (physical file path, original filename, type="file\_pointer") in the SQL database record associated with the ID returned by `mem0.add()`.
    *   `get_last_memory` would fetch the latest record from `mem0.get_all()`, check its SQL metadata, and if it's a file pointer, read the physical file.

2.  **Problem Encountered:**
    *   **`mem0.add()` returns `{'results': []}`:** The `mem0` library frequently failed to create a memory record for the descriptive pointer strings, returning an empty `results` list. This meant no ID was generated by `mem0`, so no corresponding SQL record could be reliably created for the pointer, orphaning the physical file.
    *   Database constraint errors (`NOT NULL constraint failed: memory_status_history.old_state`) also appeared initially, which were fixed by ensuring `old_state` was always populated (e.g., with a default like `MemoryState.deleted` for new entries if the schema required it).

## Iteration 2: Making Pointer Text More Unique for `mem0.add()`

1.  **Hypothesis:** `mem0` might be de-duplicating or ignoring pointer texts it deemed too similar to previous ones.
2.  **Changes to `add_memories`:**
    *   The pointer text passed to `memory_client.add()` was made more unique by embedding a UUID and a high-resolution ISO timestamp (e.g., `"File Pointer Ref: <uuid> (created <timestamp>) for <filename.md>..."`).
    *   The metadata passed to `memory_client.add()` was minimized to avoid confusing `mem0`.
    *   The full, rich file pointer metadata was primarily stored in the SQL database, linked to the ID `mem0` would hopefully provide.

3.  **Problem Persisted:**
    *   Even with highly unique pointer texts, `memory_client.add()` often still returned `{'results': []}`. Log analysis (including `mem0`'s internal "NOOP for Memory" logs) suggested `mem0` has internal heuristics that were still preventing these pointer texts from being stored as new memories.

## Iteration 3: Reverting to a Known "Working" Commit (`68859c3`)

1.  **Action:** The `openmemory/api/app/mcp_server.py` file was reverted to commit `68859c3` ("large files works"), and the Docker environment (volumes, DB) was completely reset.
2.  **Observation:**
    *   The `add_memories` in this version used a descriptive pointer string (e.g., `"File stored: {filename}. Original char length: {len}."`) without the embedded UUID/timestamp in the text itself.
    *   **Crucially, in a clean environment, `memory_client.add()` for these descriptive pointers *did* start returning valid `results` with IDs.** This allowed the SQL records (with full file metadata) and history to be created correctly.
    *   `get_last_memory` (also from `68859c3`), which relied on `mem0.get_all()` and then checked SQL metadata for file pointers, could then successfully retrieve the file content.

3.  **Conclusion for this phase:** A completely clean data environment seemed critical. The specific pointer text format in `68859c3` was acceptable to `mem0.add()` in this clean state. The failures with more "unique" pointer texts later might have been compounded by data inconsistencies from previous failed attempts or subtle changes in how `mem0` reacted to those specific string formats.

## Iteration 4 (Refinement): Ensuring `.md` Extension (Applied on top of `68859c3` principles)

1.  **Requirement:** Ensure all saved files (whether `original_filename` is provided with a different extension or not) are stored with a `.md` extension and their metadata reflects this.
2.  **Change to `add_memories`:**
    *   Logic was added to take the stem of any provided `original_filename` (or use a default like "large\_file\_content") and enforce a `.md` extension for the `actual_original_filename_as_md` used in metadata and for the physical `stored_filename_on_disk`.
    *   Added `content_type: "text/markdown"` to the SQL metadata.

## Iteration 5 (Current Pragmatic Solution): Filesystem-First `get_last_memory`

1.  **Problem Re-evaluation:** While reverting to `68859c3` showed promise, the underlying unreliability of `mem0.add()` for file pointers remained a concern for long-term stability if the "clean environment" sensitivity was high or if `mem0` behavior changed slightly. The primary goal was a very reliable way for `get_last_memory` to retrieve the latest large document intended for the Claude-to-Cursor workflow.

2.  **"Option B" Implemented:**
    *   **`add_memories`:**
        *   Continues to physically save large texts/files to disk. Filenames now include a `YYYYMMDDHHMMSSffffff` timestamp prefix and a UUID (e.g., `<timestamp>_<uuid>_originalFile.md`) for reliable chronological sorting from the filesystem.
        *   *Still attempts* to call `memory_client.add()` with an ultra-simple placeholder text (e.g., `OPENMEMORY_FILE_PLACEHOLDER_V2:<uuid>`) and minimal metadata. The success of this `mem0.add()` call is logged but no longer strictly critical for the file to be retrievable by the *primary path* of `get_last_memory`.
        *   If `mem0.add()` *does* succeed, the SQL record is created with full authoritative metadata for the file.
    *   **`get_last_memory` (Filesystem-First):**
        *   **Primary Logic:** Scans the `user_files_storage/<user_id>/` directory. Identifies the most recent file based on the timestamp in its filename. Reads and returns its content. This makes retrieval of the latest *physically saved file* very direct and independent of `mem0.add()` success for the pointer.
        *   **Fallback Logic:** If no files are found via the filesystem scan (or if an error occurs during that phase), it then falls back to querying `mem0.get_all()` to find the latest memory record known to `mem0` (which could be a short text memory or a file pointer *if* `mem0.add()` had succeeded for it). This part retains the logic of checking SQL metadata for file pointer details.

3.  **Outcome:**
    *   This provides a robust way to retrieve the latest saved document via `get_last_memory`, fulfilling the primary use case.
    *   It acknowledges that `mem0.add()` might not always create a memory for our file pointers, but we ensure the file is saved physically regardless.
    *   Short text memories are still handled through `mem0`.

## Current Status & Path Forward

The system now prioritizes reliably saving and then retrieving the latest physical file for the `get_last_memory` tool. The `add_memories` tool robustly saves the file to disk and attempts to register a simple placeholder with `mem0`.

Further refinements could include:
*   Restoring full permission-checking logic to `list_memories` and `search_memory` and deciding how/if they should incorporate knowledge of files only tracked via filesystem (if their `mem0` placeholder wasn't created).
*   Implementing robust deletion mechanisms for physical files when their (SQL-backed) pointers are deleted.
*   Continued investigation or monitoring of `mem0` library behavior regarding `add()` for various inputs.

This iterative process, involving log analysis, hypothesis testing, and targeted code changes, has led to the current, more resilient solution for the primary workflow. 