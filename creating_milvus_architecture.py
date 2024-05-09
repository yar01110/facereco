"""
from pymilvus import MilvusClient, DataType , FieldSchema,CollectionSchema,connections, db

conn = connections.connect(host="127.0.0.1", port=19530)

#database = db.create_database("Access_Management")

db.using_database("Access_Management")

# 1. Set up a Milvus client
client = MilvusClient(
    uri="http://localhost:19530",
    db_name="Access_Management"
    
)
fields = [
  FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
  FieldSchema(name="name", dtype=DataType.VARCHAR,max_length=256,  description="employee name"),
  FieldSchema(name="role", dtype=DataType.VARCHAR,max_length=256,default_value="employee",  description="employee role"),
  FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512, description="facial recognition embeddings")
]
schema = CollectionSchema(fields=fields, auto_id=True, enable_dynamic_field=True, description="employees face biometric database ")
client.list_collections()

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding", 
    index_type="IVF_FLAT",
    metric_type="L2",
    params={ "nlist": 128 }
)

client.create_collection(
    collection_name="employees",
    schema=schema,
    index_params=index_params,
    consistency_level="Eventually",
)

res = client.describe_collection(
    collection_name="employees"
)

client.load_collection("employees")
print(res)
"""