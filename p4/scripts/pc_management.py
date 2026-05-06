"""Manage Pinecone vector db here"""

from infrastructure.instances import _get_index


def main() -> None:
    """Clear all data in the fact-check-sources namespace."""
    index = _get_index()

    try:
        index.delete(delete_all=True, namespace="fact-check-sources")
        print(index.describe_index_stats())
    except Exception as e:
        print(f"Error deleting from Pinecone namespace: {e}")


if __name__ == "__main__":
    main()
