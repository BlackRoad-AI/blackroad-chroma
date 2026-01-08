from time import sleep
from chromadb.api import ClientAPI
from chromadb.test.utils.wait_for_version_increase import (
    get_collection_version,
    wait_for_version_increase,
)


def test_indexing_status_empty_collection(client: ClientAPI) -> None:
    """Test indexing status on empty collection"""
    client.reset()

    collection = client.create_collection(name="test_collection")
    status = collection.get_indexing_status()

    assert isinstance(status, dict)
    assert status["num_indexed_ops"] == 0
    assert status["num_unindexed_ops"] == 0
    assert status["total_ops"] == 0
    assert status["indexing_progress"] == 1.0


def test_indexing_status_after_add(client: ClientAPI) -> None:
    """Test indexing status after adding embeddings"""
    client.reset()

    collection = client.create_collection(name="test_collection")

    ids = [f"id_{i}" for i in range(300)]
    embeddings = [[float(i), float(i + 1), float(i + 2)] for i in range(300)]
    initial_version = get_collection_version(client, collection.name)
    collection.add(ids=ids, embeddings=embeddings)  # type: ignore

    status = collection.get_indexing_status()
    assert status["total_ops"] == 300

    if initial_version == get_collection_version(client, collection.name):
        assert isinstance(status, dict)
        assert status["num_unindexed_ops"] == 300
        assert status["num_indexed_ops"] == 0
        assert status["indexing_progress"] == 0.0
        wait_for_version_increase(client, collection.name, initial_version)
        # Give some time to invalidate the frontend query cache
        sleep(60)

        # Check status after indexing completes
        final_status = collection.get_indexing_status()
        assert isinstance(final_status, dict)
        assert final_status["num_indexed_ops"] == 300
        assert final_status["num_unindexed_ops"] == 0
        assert final_status["indexing_progress"] == 1.0


def test_indexing_status_after_upsert(client: ClientAPI) -> None:
    """Test indexing status after upsert operations"""
    client.reset()

    collection = client.create_collection(name="test_collection")
    initial_version = get_collection_version(client, collection.name)

    collection.upsert(ids=["id1", "id2"], embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # type: ignore

    status = collection.get_indexing_status()
    assert status["total_ops"] == 2

    if initial_version == get_collection_version(client, collection.name):
        assert isinstance(status, dict)
        assert status["num_unindexed_ops"] == 2
        assert status["num_indexed_ops"] == 0
        assert status["indexing_progress"] == 0.0
        wait_for_version_increase(client, collection.name, initial_version)
        sleep(60)

    collection.upsert(ids=["id1", "id3"], embeddings=[[1.1, 2.1, 3.1], [7.0, 8.0, 9.0]])  # type: ignore

    status = collection.get_indexing_status()
    assert status["total_ops"] == 4


def test_indexing_status_after_delete(client: ClientAPI) -> None:
    """Test indexing status after delete operations"""
    client.reset()

    collection = client.create_collection(name="test_collection")
    initial_version = get_collection_version(client, collection.name)

    collection.add(
        ids=["id1", "id2", "id3"],
        embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],  # type: ignore
    )

    if initial_version == get_collection_version(client, collection.name):
        status = collection.get_indexing_status()
        assert isinstance(status, dict)
        assert status["num_unindexed_ops"] == 3
        assert status["num_indexed_ops"] == 0
        assert status["indexing_progress"] == 0.0
        wait_for_version_increase(client, collection.name, initial_version)
        sleep(60)

    initial_status = collection.get_indexing_status()
    assert initial_status["total_ops"] == 3

    collection.delete(ids=["id1", "id2"])

    # Delete adds operations to the log, so total_ops increases
    status_after_delete = collection.get_indexing_status()
    assert status_after_delete["total_ops"] == 5


def test_indexing_status_field_types(client: ClientAPI) -> None:
    """Test that indexing status returns correct field types"""
    client.reset()

    collection = client.create_collection(name="field_types_collection")
    initial_version = get_collection_version(client, collection.name)

    collection.add(ids=["type_test_id"], embeddings=[[1.0, 2.0, 3.0]])  # type: ignore

    status = collection.get_indexing_status()

    if initial_version == get_collection_version(client, collection.name):
        assert isinstance(status, dict)
        assert status["num_unindexed_ops"] == 1
        assert status["num_indexed_ops"] == 0
        assert status["indexing_progress"] == 0.0
        wait_for_version_increase(client, collection.name, initial_version)
        sleep(60)

    final_status = collection.get_indexing_status()

    assert isinstance(final_status["num_indexed_ops"], int)
    assert isinstance(final_status["num_unindexed_ops"], int)
    assert isinstance(final_status["total_ops"], int)
    assert isinstance(final_status["indexing_progress"], float)

    assert final_status["num_indexed_ops"] >= 0
    assert final_status["num_unindexed_ops"] >= 0
    assert final_status["total_ops"] >= 0
    assert 0.0 <= final_status["indexing_progress"] <= 1.0


def test_indexing_status_not_found(client: ClientAPI) -> None:
    """Test indexing status on non-existent collection"""
    client.reset()

    collection = client.create_collection(name="temp_collection")
    client.delete_collection("temp_collection")

    try:
        collection.get_indexing_status()
        assert False, "Expected exception for non-existent collection"
    except Exception as e:
        assert (
            "not found" in str(e).lower()
            or "does not exist" in str(e).lower()
            or "soft deleted" in str(e).lower()
        )
