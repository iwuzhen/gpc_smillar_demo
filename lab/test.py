import redis
import time
import uuid
import argparse
import pymongo
import tqdm
import functools
import psycopg
import networkx as nx
import requests
import json

connection_pool = redis.ConnectionPool(host='192.168.1.227', port=6379, db=0)

# 创建 Redis 客户端实例，使用上面创建的连接池
r = redis.Redis(connection_pool=connection_pool,decode_responses=True)

# stream_name = "test_stream"
consumer_name = f"consumer-{uuid.uuid4()}"
group_name = 'my_group'
collection_name = "token_similarity_v20240729"

def init_xgroup(stream_name, group_name):
    try:
        r.xgroup_create(stream_name, group_name,id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        # 如果组已经存在，则忽略错误
        if "BUSYGROUP Consumer Group name already exists" not in str(e):
            raise
        
        # if "BUSYGROUP" in str(e):
        #     print(f"Group '{group_name}' already exists, deleting and recreating it.")
            
        #     # 删除现有的消费者组
        #     r.xgroup_destroy(stream_name, group_name)
        #     print(f"Group '{group_name}' deleted.")
            
        #     # 重新创建消费者组
        #     r.xgroup_create(stream_name, group_name, mkstream=True)
        #     print(f"Group '{group_name}' recreated successfully.")
        # else:
        #     # 处理其他错误
        #     print(f"Error creating group: {e}")
        
    
session = requests.Session()
url = 'http://192.168.1.227:10004/wiki_api/api/tree/editDistance'


def get_max_connected_graph(edges_with_weight):
    G = nx.Graph()
    G.add_weighted_edges_from(edges_with_weight)
    connected_components = list(nx.connected_components(G))
    max_component = max(connected_components, key=len)
    max_subgraph = G.subgraph(max_component)
    
    # 计算最小生成树
    mst = nx.minimum_spanning_tree(max_subgraph)

    connected_graph = []
    for edge in mst.edges(data=False):
        connected_graph.append(edge)
    return connected_graph

def get_tree_editance(struct_1,struct_2, query_distance):
    ret_data = {}
    
    connected_edges_1 = query_distance(struct_1,struct_1)
    ret_data['connected_tokens_1'] = len(connected_edges_1)
    if connected_edges_1:
        connected_edges_1 = get_max_connected_graph([(x,y,v) for x,y,v in connected_edges_1])

    connected_edges_2 = query_distance(struct_2,struct_2)
    ret_data['connected_tokens_2'] = len(connected_edges_2)
    if connected_edges_2:
        connected_edges_2 = get_max_connected_graph([(x,y,v) for x,y,v in connected_edges_2])
        
    if not connected_edges_2 or not connected_edges_1:
        ret_data['tree_edit_distance'] = 255
        return ret_data
    
    # flag = 0, 只考虑结构
    graph_1 = [{'title':'a','edge':connected_edges_1}]
    graph_2 = [{'title':'b','edge':connected_edges_2}]
    struct_req_data = {
        "stra": json.dumps(graph_1),
        "strb": json.dumps(graph_2),
        "flag":0
    }
    try:
        rep = session.post(url, json=struct_req_data, timeout=1)
    except requests.exceptions.ReadTimeout as e:
        print(struct_req_data)
        raise e
    tree_edit_distance = rep.json().get("data",{}).get('a_b')
    # print(graph_1)
    # print(graph_2)
    # print(rep.json())
    ret_data['tree_edit_distance'] = tree_edit_distance
    if not tree_edit_distance:
        print('struct_req_data', rep.json(), struct_req_data)
    return ret_data
        
                

def generate_message(group_name, consumer_name, stream_name):
    print("start generate message", group_name,stream_name, consumer_name)
    while True:
        try:
            messages = r.xreadgroup(group_name, consumer_name, {stream_name: '>'}, count=1, block=20000)
            if not messages:
                print("No new messages")
                # break
            for stream, msgs in messages:
                for msg_id, msg_data in msgs:
                    yield msg_id, msg_data
                    # print(f"Message ID: {msg_id}, Message Data: {msg_data}")
                    # time.sleep(1)
                    # 在处理完消息后，可以确认消息已被处理
                    # r.xack(stream_name, group_name, msg_id)
        except Exception as e:
            print(f"Error reading messages: {e}")
            break

# 任务0, 生成第一个消息队列, 
def task_0_init_queue():
    stream_name = "task_0"
    
    DATABASE = pymongo.MongoClient("192.168.1.222").temporary_token_similarity
    Collection = DATABASE[collection_name]
    
    stream_length = r.xlen(stream_name)
    print(f"Stream '{stream_name}' 的消息长度为: {stream_length}")
    r.delete(stream_name)
    init_xgroup(stream_name, group_name)
    stream_length = r.xlen(stream_name)
    print(f"Stream '{stream_name}' 的消息长度为: {stream_length}")
    import random
    doc_list = []
    for i, doc in tqdm.tqdm(enumerate(Collection.find({'v1':None,'status':None,'weight':{'$lte':1}}).sort("weight", pymongo.DESCENDING))): 
        # todo 测试
        # if i < 10000:
        #     continue
        # if i > 10010:
        #     break
        doc_list.append(doc)
        
    random.shuffle(doc_list)
    for doc in doc_list:
        r.xadd(stream_name, doc)

    stream_length = r.xlen(stream_name)
    print(f"Stream '{stream_name}' 的消息长度为: {stream_length}")


# 过滤和找齐 plain text    
def task_1_get_plaintext():
    import sys
    sys.path.append("gpc_demo")
    from utils import (
        get_plaintext, 
    )    
    get_plaintext = functools.lru_cache(maxsize=1000_000)(get_plaintext)
    
    DATABASE = pymongo.MongoClient("192.168.1.222").temporary_token_similarity
    Collection = DATABASE[collection_name]

    last_stream_name = "task_0"
    current_stream_name = "task_1"
    r.delete(current_stream_name)
    # init_xgroup(current_stream_name, group_name)
    init_xgroup(current_stream_name, group_name)
    
    for msg_id, msg_data in tqdm.tqdm(generate_message(group_name, consumer_name, last_stream_name)):
        ID = msg_data[b'_id'].decode()
        # r.xack(last_stream_name, group_name, msg_id)
        # continue
        # print(ID)
        # continue
        
        plaintext_b = get_plaintext(msg_data[b'EID'].decode())
        plaintext_a = get_plaintext(msg_data[b'SID'].decode())
        
        if not plaintext_a or not plaintext_b:
            print('text miss')
            Collection.update_one({'_id': ID}, {'$set': {'status': 'text miss'}})
            continue
    
        if len(plaintext_a.split(" "))<=15 or len(plaintext_b.split(" "))<=15:
            print('text short', end=" ")
            Collection.update_one({'_id': ID}, {'$set': {'status': 'text short'}})
            continue
    
        msg_data['plaintext_b'] = plaintext_b
        msg_data['plaintext_a'] = plaintext_a
        
        r.xadd(current_stream_name, msg_data)
        r.xack(last_stream_name, group_name, msg_id)
    

# token 计算    
def task_2_token_anlysis():
    import sys
    sys.path.append("gpc_demo")
    from utils import (
        calculate_cartesian_product_similarity_pgs, 
        get_token,
        query_distance
    )

    
    DATABASE = pymongo.MongoClient("192.168.1.222").temporary_token_similarity
    Collection = DATABASE[collection_name]

    last_stream_name = "task_1"
    current_stream_name = "task_2"
    init_xgroup(current_stream_name, group_name)
    
    for msg_id, msg_data in tqdm.tqdm(generate_message(group_name, consumer_name, last_stream_name)):
        ID = msg_data[b'_id'].decode()
        plaintext_a = msg_data[b'plaintext_a'].decode()
        plaintext_b = msg_data[b'plaintext_b'].decode()
        
        token_set_1, _ = get_token(plaintext_a)
        token_set_2, _ = get_token(plaintext_b)
        
        df, _ = calculate_cartesian_product_similarity_pgs(
            tuple(token_set_1 - token_set_2), 
            tuple(token_set_2 - token_set_1),
            tuple(token_set_2 & token_set_1),
        )

        if not len(df):
            print('lost tokens')
            Collection.update_one({'_id': ID}, {'$set': {'status': 'lost tokens'}})
            continue

        msg_data['token_similarity'] = df['similarity'].mean()
        
        # 计算结构相似度
        struct_1 = tuple(item[0] for item in token_set_1)
        struct_2 = tuple(item[0] for item in token_set_2)
        
        # {'connected_tokens_1': 6, 'connected_tokens_2': 0, 'tree_edit_distance': 255}
        # print(struct_1,struct_2)
        tree_edit_diance_report = get_tree_editance(struct_1, struct_2, query_distance)
        msg_data.update(tree_edit_diance_report)
    
        # print(msg_data)
        try:
            r.xadd(current_stream_name, msg_data)
        except redis.exceptions.DataError as e:
            print(msg_data)
            raise e
            
        r.xack(last_stream_name, group_name, msg_id)
    
    
# sts 计算    
def task_3_sts_anlysis():
    from scipy.spatial.distance import (
        cosine,
        euclidean,
        minkowski,
        cityblock,
        chebyshev,
    )
    
    from modelscope import AutoModel
    JinNaAI_Model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
    
    DATABASE = pymongo.MongoClient("192.168.1.222").temporary_token_similarity
    Collection = DATABASE[collection_name]

    last_stream_name = "task_2"
    
    for msg_id, msg_data in tqdm.tqdm(generate_message(group_name, consumer_name, last_stream_name)):
        ID = msg_data[b'_id'].decode()
        plaintext_a = msg_data[b'plaintext_a'].decode()
        plaintext_b = msg_data[b'plaintext_b'].decode()
        
        
        token_similarity =  float(msg_data[b'token_similarity'])
        
        # {'connected_tokens_1': 6, 'connected_tokens_2': 0, 'tree_edit_distance': 255}
        connected_tokens_1 =  int(msg_data[b'connected_tokens_1'])
        connected_tokens_2 =  int(msg_data[b'connected_tokens_2'])
        tree_edit_distance =  int(msg_data[b'tree_edit_distance'])
        
        embeddings = JinNaAI_Model.encode([plaintext_a, plaintext_b])
        cosine_similarity = cosine(embeddings[0],embeddings[1])
        euclidean_similarity = euclidean(embeddings[0],embeddings[1])
        minkowski_similarity = minkowski(embeddings[0],embeddings[1])
        cityblock_similarity = cityblock(embeddings[0],embeddings[1])
        chebyshev_similarity = chebyshev(embeddings[0],embeddings[1])
        Collection.update_one({'_id': ID},{"$set":{
            'v1':{
                'token':token_similarity,
                'cosine': cosine_similarity.item(),
                'euclidean': euclidean_similarity,
                'minkowski': minkowski_similarity,
                'cityblock': cityblock_similarity.item(),
                'chebyshev': chebyshev_similarity.item(),
                'connected_tokens_1': connected_tokens_1,
                'connected_tokens_2': connected_tokens_2,
                'tree_edit_distance': tree_edit_distance,
            }
        }})
        # print("reslt")
        # print({
        #     'v1':{
        #         'token':token_similarity,
        #         'cosine': cosine_similarity.item(),
        #         'euclidean': euclidean_similarity,
        #         'minkowski': minkowski_similarity,
        #         'cityblock': cityblock_similarity.item(),
        #         'chebyshev': chebyshev_similarity.item(),
        #     }
        # })
            
    
        
        r.xack(last_stream_name, group_name, msg_id)
        # return
    

def test():
    import sys
    import requests
    sys.path.append("gpc_demo")
    import time
    from utils import (
        query_distance
    )
    ret = get_tree_editance((123131,51235214,27298083,36645032,18963910 ),
                            (18963910 ,123131,51235214,27298083,36645032),
                            query_distance)
    print(ret)
    
    
    import sys
    sys.path.append("gpc_demo")
    from utils import (
        calculate_cartesian_product_similarity_pgs, 
        get_token,
        query_distance
    )
    last_stream_name = "task_1"
    current_stream_name = "task_2"
    init_xgroup(current_stream_name, group_name)
    
    for msg_id, msg_data in tqdm.tqdm(generate_message(group_name, consumer_name, last_stream_name)):
        
        # a = time.time()
        ID = msg_data[b'_id'].decode()
        plaintext_a = msg_data[b'plaintext_a'].decode()
        plaintext_b = msg_data[b'plaintext_b'].decode()
        
        a = time.time()
        token_set_1, _ = get_token(plaintext_a)
        token_set_2, _ = get_token(plaintext_b)
        
        print("a time", time.time() -a)
        
        df, _ = calculate_cartesian_product_similarity_pgs(
            tuple(token_set_1 - token_set_2), 
            tuple(token_set_2 - token_set_1),
            tuple(token_set_2 & token_set_1),
        )


        msg_data['token_similarity'] = df['similarity'].mean()
        print("a time", time.time() -a)
        a = time.time()
        # 计算结构相似度
        struct_1 = (item[0] for item in token_set_1)
        struct_2 = (item[0] for item in token_set_2)
        
        # {'connected_tokens_1': 6, 'connected_tokens_2': 0, 'tree_edit_distance': 255}
        tree_edit_diance_report = get_tree_editance(struct_1, struct_2, query_distance)
        msg_data.update(tree_edit_diance_report)
        
        print("b time", time.time() -a)
        
        return
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="this is help")
    parser.add_argument('method', type=int, 
                        choices=[-1, 0,1,2,3,4,5,6,7,8,9], 
                        help="this is help")

    args = parser.parse_args()

    if args.method == -1:
        test()
    if args.method == 0:
        task_0_init_queue()
    elif args.method == 1:
        task_1_get_plaintext()
    elif args.method == 2:
        task_2_token_anlysis()
    elif args.method == 3:
        task_3_sts_anlysis()