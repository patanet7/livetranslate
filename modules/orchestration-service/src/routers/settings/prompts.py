"""
Prompts Settings Router

Handles prompt template CRUD operations, testing, performance analysis,
comparison, and management endpoints.
"""

from ._shared import (
    TRANSLATION_SERVICE_URL,
    Any,
    APIRouter,
    HTTPException,
    Optional,
    PromptTemplateRequest,
    PromptTestRequest,
    aiohttp,
    get_translation_service_client,
    logger,
)

router = APIRouter(tags=["settings-prompts"])


# ============================================================================
# Prompt Management Endpoints
# ============================================================================


@router.get("/prompts")
async def get_prompts(
    active: Optional[bool] = None,
    category: Optional[str] = None,
    language_pair: Optional[str] = None,
):
    """Get all prompt templates with optional filtering"""
    try:
        async with await get_translation_service_client() as client:
            params = {}
            if active is not None:
                params["active"] = "true" if active else "false"
            if category:
                params["category"] = category
            if language_pair:
                params["language_pair"] = language_pair

            async with client.get(f"{TRANSLATION_SERVICE_URL}/prompts", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "prompts": data.get("prompts", []),
                        "total_count": data.get("total_count", 0),
                        "filters_applied": data.get("filters_applied", {}),
                        "message": "Prompts retrieved successfully",
                    }
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prompts: {e!s}") from e


@router.get("/prompts/statistics")
async def get_prompt_statistics():
    """Get overall prompt management statistics"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/prompts/statistics") as response,
        ):
            if response.status == 200:
                stats = await response.json()
                return {
                    "success": True,
                    "statistics": stats,
                    "message": "Prompt statistics retrieved successfully",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve prompt statistics: {e!s}"
        ) from e


@router.get("/prompts/categories")
async def get_prompt_categories():
    """Get available prompt categories"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/prompts/categories") as response,
        ):
            if response.status == 200:
                categories = await response.json()
                return {
                    "success": True,
                    "categories": categories.get("categories", []),
                    "total_count": categories.get("total_count", 0),
                    "message": "Prompt categories retrieved successfully",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt categories: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve prompt categories: {e!s}"
        ) from e


@router.get("/prompts/variables")
async def get_prompt_variables():
    """Get available prompt template variables"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/prompts/variables") as response,
        ):
            if response.status == 200:
                variables = await response.json()
                return {
                    "success": True,
                    "variables": variables.get("variables", []),
                    "usage_example": variables.get("usage_example", ""),
                    "message": "Prompt variables retrieved successfully",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt variables: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve prompt variables: {e!s}"
        ) from e


@router.post("/prompts/compare")
async def compare_prompts(comparison_request: dict[str, Any]):
    """Compare performance of multiple prompts"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.post(
                f"{TRANSLATION_SERVICE_URL}/prompts/compare", json=comparison_request
            ) as response,
        ):
            if response.status == 200:
                comparison = await response.json()
                return {
                    "success": True,
                    "comparison_results": comparison,
                    "message": "Prompt comparison completed successfully",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to compare prompts: {e!s}") from e


@router.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str):
    """Get a specific prompt template"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}") as response,
        ):
            if response.status == 200:
                prompt_data = await response.json()
                return {
                    "success": True,
                    "prompt": prompt_data,
                    "message": "Prompt retrieved successfully",
                }
            elif response.status == 404:
                raise HTTPException(status_code=404, detail="Prompt not found")
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prompt: {e!s}") from e


@router.post("/prompts")
async def create_prompt(prompt: PromptTemplateRequest):
    """Create a new prompt template"""
    try:
        async with await get_translation_service_client() as client:
            prompt_data = prompt.dict()
            async with client.post(
                f"{TRANSLATION_SERVICE_URL}/prompts", json=prompt_data
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return {
                        "success": True,
                        "prompt_id": result.get("prompt_id"),
                        "message": "Prompt created successfully",
                    }
                elif response.status == 409:
                    raise HTTPException(
                        status_code=409, detail="Prompt with this ID already exists"
                    )
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create prompt: {e!s}") from e


@router.put("/prompts/{prompt_id}")
async def update_prompt(prompt_id: str, updates: dict[str, Any]):
    """Update an existing prompt template"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.put(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}", json=updates) as response,
        ):
            if response.status == 200:
                await response.json()
                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "message": "Prompt updated successfully",
                }
            elif response.status == 404:
                raise HTTPException(status_code=404, detail="Prompt not found")
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update prompt: {e!s}") from e


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str):
    """Delete a prompt template"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.delete(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}") as response,
        ):
            if response.status == 200:
                await response.json()
                return {
                    "success": True,
                    "prompt_id": prompt_id,
                    "message": "Prompt deleted successfully",
                }
            elif response.status == 404:
                raise HTTPException(
                    status_code=404,
                    detail="Prompt not found or cannot be deleted (default prompts are protected)",
                )
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {e!s}") from e


@router.post("/prompts/{prompt_id}/test")
async def test_prompt(prompt_id: str, test_data: PromptTestRequest):
    """Test a prompt template with sample data"""
    try:
        async with await get_translation_service_client() as client:
            test_payload = test_data.dict()
            async with client.post(
                f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}/test", json=test_payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "test_result": result.get("test_result"),
                        "prompt_used": result.get("prompt_used"),
                        "system_message": result.get("system_message"),
                        "prompt_analysis": result.get("prompt_analysis"),
                        "message": "Prompt test completed successfully",
                    }
                elif response.status == 404:
                    raise HTTPException(status_code=404, detail="Prompt not found")
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Translation service error: {error_text}",
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing prompt {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test prompt: {e!s}") from e


@router.get("/prompts/{prompt_id}/performance")
async def get_prompt_performance(prompt_id: str):
    """Get performance analysis for a prompt"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.get(f"{TRANSLATION_SERVICE_URL}/prompts/{prompt_id}/performance") as response,
        ):
            if response.status == 200:
                analysis = await response.json()
                return {
                    "success": True,
                    "performance_analysis": analysis,
                    "message": "Performance analysis retrieved successfully",
                }
            elif response.status == 404:
                raise HTTPException(status_code=404, detail="Prompt not found")
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving performance analysis for prompt {prompt_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve performance analysis: {e!s}"
        ) from e


@router.post("/translation/test")
async def test_translation_with_prompt(translation_request: dict[str, Any]):
    """Test translation using a specific prompt template"""
    try:
        async with (
            await get_translation_service_client() as client,
            client.post(
                f"{TRANSLATION_SERVICE_URL}/translate/with_prompt",
                json=translation_request,
            ) as response,
        ):
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "translation_result": {
                        "translated_text": result.get("translated_text"),
                        "source_language": result.get("source_language"),
                        "target_language": result.get("target_language"),
                        "confidence_score": result.get("confidence_score"),
                        "processing_time": result.get("processing_time"),
                        "backend_used": result.get("backend_used"),
                        "prompt_id": result.get("prompt_id"),
                        "prompt_used": result.get("prompt_used"),
                    },
                    "message": "Translation test completed successfully",
                }
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Translation service error: {error_text}",
                )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to translation service: {e}")
        raise HTTPException(status_code=503, detail="Translation service unavailable") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing translation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test translation: {e!s}") from e
