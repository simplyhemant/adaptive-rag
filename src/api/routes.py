"""
API routes for RAG operations.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from src.memory.chat_history_mongo import ChatHistory
from src.memory.chathistory_in_memory import ChatInMemoryHistory
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/rag/query")
async def rag_query(req: QueryRequest):
    """
    Process a RAG query and return the result.

    Args:
        req: The query request containing query text and session_id.

    Returns:
        The generated response from the RAG pipeline.
    """
    try:
        try:
            chat_history = ChatHistory.get_session_history(req.session_id)
            await chat_history.add_message(HumanMessage(content=req.query))
            messages = await chat_history.get_messages()
            use_mongo = True
        except Exception as e:
            logger.warning("MongoDB unavailable, falling back to in-memory history: %s", e)
            chat_history = ChatInMemoryHistory.get_session_history(req.session_id)
            chat_history.add_message(HumanMessage(content=req.query))
            messages = chat_history.messages
            use_mongo = False

        result = builder.invoke({
            "messages": messages,
            "latest_query": req.query,
            "rewrite_count": 0,
        })
        output_text = result["messages"][-1].content

        # Save assistant message
        if use_mongo:
            try:
                await chat_history.add_message(AIMessage(content=output_text))
            except Exception as e:
                logger.warning("Failed to save assistant message to MongoDB: %s", e)
        else:
            chat_history.add_message(AIMessage(content=output_text))

        return {"result": result["messages"][-1]}
    except Exception as e:
        logger.exception("Error in /rag/query: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/rag/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(..., alias="X-Description")
):
    """
    Upload a document for RAG processing.

    Args:
        file: The file to upload (PDF or TXT).
        description: Document description provided via header.

    Returns:
        Upload status.
    """
    try:
        status_upload = documents(description, file)
        return {"status": status_upload}
    except Exception as e:
        logger.exception("Error in /rag/documents/upload: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )

