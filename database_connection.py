from pymilvus import connections , db ,MilvusClient

#creating the database
connections.connect(host="127.0.0.1", port=19530,db_name="Access_Management")




def connectionToDb(database="Access_Management",collec="employees"):
    
    connections.connect(
        host="127.0.0.1",
        port=19530,
        db_name="Access_Management"
        )
    
    client = MilvusClient(
     uri="http://localhost:19530",
     db_name="Access_Management"
    )
    client.load_collection(collec)
    return client

def researchVector_ById(client:MilvusClient,vector_list,collection="employees"):
    res = client.search(
    collection_name="employees", # Replace with the actual name of your collection
    data=[
        *vector_list
    ], # Replace with your query vectors
    limit=1, # Max. number of search results to return
    search_params={"metric_type": "L2", "params": {}}) 
    return res

def get_names_from_embeddings(client:MilvusClient,vector_list,collection="employees",threshold=0.5):
    
    res=researchVector_ById(client,vector_list=vector_list,collection=collection)
    if len(res[0])==0:
        return 0
    def get_nameandrole_ById(client,ids):
        res = client.get(
        collection_name="employees",
        ids=ids
        )
        return {"name":res[0].get("name"),"role":res[0].get("role")}
    
    identification = [get_nameandrole_ById(client, id[0].get('id')) if id[0].get('distance') < threshold else "unidentified" for id in res]
    
    return identification

    
