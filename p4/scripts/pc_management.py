from infrastructure.instances import _get_index

def main():
    index = _get_index()
    
    try:
        index.delete(delete_all=True, namespace="fact-check-sources")
        print(index.describe_index_stats())
    except:
        print("Error deleting from Pinecone namespace")

if __name__ == "__main__":
    main()
