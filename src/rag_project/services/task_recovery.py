from rag_project.services.store import Store


INTERRUPTED_TASK_STATUSES = {"pending", "running"}
INTERRUPTED_DOCUMENT_STATUSES = {"parsing", "chunking", "embedding"}
INTERRUPTED_TASK_ERROR = "Task was interrupted by an application restart. Re-run the operation."


async def mark_interrupted_tasks_failed(store: Store) -> int:
    """Fail tasks left active by a previous process.

    FastAPI BackgroundTasks are in-process. After a container rebuild or app restart,
    existing background coroutines are gone even though their database rows remain.
    """
    count = 0
    for task in list(store.tasks.values()):
        if task.status not in INTERRUPTED_TASK_STATUSES:
            continue
        await store.update_task(task.task_id, status="failed", error=INTERRUPTED_TASK_ERROR)
        count += 1
        if not task.document_id:
            continue
        document = store.get_document(task.document_id)
        if document is not None and document.status in INTERRUPTED_DOCUMENT_STATUSES:
            await store.update_document(task.document_id, status="failed", error=INTERRUPTED_TASK_ERROR)
    return count
