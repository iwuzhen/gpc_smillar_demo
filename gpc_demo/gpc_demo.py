import streamlit as st
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import json
import requests
from gpc_demo.utils import (
    calculate_cartesian_product_similarity_pgs, 
    calculate_cartesian_product_similarity,
    get_token, 
    search_wikipedia, 
    get_plaintext, 
    calculate_similarity,
    query_partner_distancles,
    get_llm_article,
    get_tree_similarity
    )
from gpc_demo.setting import (
    TITLE_SIMIARLITY_NAME,
    STS_SIMIARLITY_NAME,
    TOKEN_SIMIARLITY_NAME,
    TREE_SIMIARLITY_NAME
)
from streamlit_searchbox import st_searchbox
from annotated_text import annotated_text

get_llm_article = st.cache_data(get_llm_article, max_entries=1_000)
calculate_cartesian_product_similarity_pgs = st.cache_data(calculate_cartesian_product_similarity_pgs, max_entries=1_000)
calculate_cartesian_product_similarity = st.cache_data(calculate_cartesian_product_similarity, max_entries=1_000_000)

get_tree_similarity = st.cache_data(get_tree_similarity, max_entries=1_000)
get_plaintext = st.cache_data(get_plaintext, max_entries=100_000)
get_token = st.cache_data(get_token, max_entries=100_000)
calculate_similarity = st.cache_data(calculate_similarity, max_entries=100_000)


title_session_key = "title_state"
if title_session_key not in st.session_state:
    tmp_data = {
        "name_ID_dict": {
            "Beijing":18603746,
            "Harvard University":18426501,
            "Mathematics": 18831,
            "Physics":22939,
            "Chemistry":5180,
            "Biology":9127632,
            "girl":229041,
            "boy":872660,
            "man":371250,
            "woman":33183,
            "King":52337107,'History':10772350,
            "Queen":25180,
            "Humans": 682482,'Sociology':18717981,
            "Male":1822282,'Materials science':19622,
            "Female":20913864,"Medicine":18957,"Computer science":5323,'Engineering':9251
        },
        "a_select":["Computer science",'Engineering','Materials science','History','Sociology', "Mathematics","Physics","Medicine", "Chemistry","Biology", "man","woman","boy", "girl", "King", "Queen","Female","Male","Humans",],
        "b_select":["Computer science",'Engineering','Materials science','History','Sociology', "Physics","Mathematics","Chemistry","Medicine","Biology", "man","woman","boy", "girl", "King", "Queen","Female","Male",  ],
    }
    tmp_data['select_option'] = set(tmp_data['name_ID_dict'].keys())
    # tmp_data['a_select'] = [v for v in tmp_data['b_select']]
    st.session_state[title_session_key] = tmp_data
    
# session_state
token_data_list_1,token_data_list_2= [],[]

st.header("GPC 句子之间的相关性",divider='rainbow')

with st.expander("方法论"):
    st.markdown("#### 方法论\n\n"
                "1. 从 wikipedia 找到词条的第一段话.\n\n"
                "2. 对这段话进行 tokenizer, 获得文本中含有的的 title, 标记出来作为 token.\n\n"
                "3. 对比两段文本中的token, 因为是文本之间的对比, 两个tokne集合需要去重.\n\n"
                "4. 两个token集合在 GPC 数据库中进行搜索, 获取 token 两两的距离, 得到一个子图. \n\n"
                "5. 不生成单个孤岛节点的情况下, 尽可能的删除距离大的边.\n\n"
                "6. 留下的子图计算平均距离.\n\n")


    st.markdown("#### 附录 相关系数: Spearman vs Pearson  \n\n"
                "###### Spearman Correlation Coefficient（ Spearman 相关系数）\n\n"
                "> Wikipedia Definition: In statistics, Spearman’s rank correlation coefficient or Spearman’s ρ, named after Charles Spearman is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a monotonic function.\n\n"
                "统计学中，以 Charles Spearman 命名的 Spearman相关系数(Spearman ρ) 是排序相关性（ranked values)的非参数度量。使用单调函数描述两个变量之间的关系的程度。\n\n"
                "重要推论：Spearman ρ 相关性可以评估两个变量之间的单调关系——有序，它基于每个变量的排名值(ranked value)而不是原始数据(original value)\n\n"
                "###### Pearson Correlation Coefficient(皮尔逊相关系数)\n\n"
                "> Wikipedia Definition: In statistics, the Pearson correlation coefficient also referred to as Pearson’s r or the bivariate correlation is a statistic that measures the linear correlation between two variables X and Y. It has a value between +1 and −1. A value of +1 is a total positive linear correlation, 0 is no linear correlation, and −1 is a total negative linear correlation. \n\n"
                "皮尔逊相关系数(r)，是衡量两个变量X和Y之间线性相关性的统计量。它的值介于 +1 和 -1 之间。+1 完全正线性相关，0 是无线性相关，-1 是完全负线性相关。\n\n"
                "重要推论：皮尔逊相关性只能评估两个连续变量之间的线性相关（仅当一个变量的变化与另一个变量的比例变化相关时，关系才是线性的） \n\n")
        
    st.markdown("""\n\n
    **相关性区间**: \n\n
    0.00到0.19：几乎无相关性。\n\n
    0.20到0.39：低度相关性。\n\n
    0.40到0.59：中度相关性。\n\n
    0.60到0.79：显著相关性。\n\n
    0.80到0.99：高度相关性。\n\n
    """)
st.divider()
# def search_wikipediaaa(searchterm: str) -> list[any]:
#     return ["a"]


# st_searchbox(
#     search_wikipediaaa,
#     label="ffc第一个词条aaa",
# )
col1, col2 = st.columns(2)

col1.markdown("**参数调整区**")
col2.markdown("**结果展示区**")


col_a, col_b = col1.columns(2)
container_a = col_a.container(border=True)
container_b = col_b.container(border=True)



with container_a:
    
    wiki_title_a_pageID = st_searchbox(
        search_wikipedia,
        label="第一个词条",
        # default="18426501",
        # default="Harvard University",
        default_use_searchterm=False,
        placeholder="search",
        key="wiki_title_a",
        debounce=200,
        min_execution_time=200,
    )
    # 选择好的词,添加进 multiselect
    if wiki_title_a_pageID:
        session_data = st.session_state['wiki_title_a']
        
        select_title = ""
        # updte title id
        for i, ID in enumerate(session_data['options_py']):
            title = session_data['options_js'][i]['label']
            st.session_state[title_session_key]['name_ID_dict'][title] = ID
            if ID == wiki_title_a_pageID:
                select_title = title
        

        if select_title:
            st.session_state[title_session_key]['select_option'].add(select_title)
            _list = set(st.session_state[title_session_key]['a_select'])
            _list.add(select_title)
            st.session_state[title_session_key]['a_select'] = list(_list)
    
    options_a = st.multiselect(
        "选中的词条",
        list(st.session_state[title_session_key]['select_option']),
        default=st.session_state[title_session_key]['a_select'],
    )
    if len(options_a) != len(st.session_state[title_session_key]['a_select']):
        # print("modify",len(st.session_state[title_session_key]['a_select']), len(options_a))
        st.session_state[title_session_key]['a_select'] = options_a

    # st.divider() 
    # with st.container(border=True):
    #     custom_title = st.text_input("自定义文案主题", "Sunshine", key="custom_title_a")
    #     with st.empty():
    #         custom_text = get_llm_article(custom_title)
    #         txt_a = st.text_area(
    #             f"自定义文案 - {custom_title}",
    #             custom_text,
    #             key="custom_area_a"
    #         )
            
    #     if custom_title != "":
    #         token_set_1, annotated_text_list = get_token(txt_a)
    #         token_data_list_1.append({
    #             'ID':0,
    #             'title':f"*{custom_title}*",
    #             'token_set':token_set_1,
    #             'plaintext': txt_a
    #         })
    
with container_b:
    wiki_title_b_pageID = st_searchbox(
        search_wikipedia,
        label="第二个词条",
        placeholder="search",
        key="wiki_title_b",
        debounce=200,
        min_execution_time=200,
    )
    # 选择好的词,添加进 multiselect
    if wiki_title_b_pageID:
        session_data = st.session_state['wiki_title_b']
        
        select_title = ""
        # updte title id
        for i, ID in enumerate(session_data['options_py']):
            title = session_data['options_js'][i]['label']
            st.session_state[title_session_key]['name_ID_dict'][title] = ID
            if ID == wiki_title_b_pageID:
                select_title = title
        

        if select_title:
            st.session_state[title_session_key]['select_option'].add(select_title)
            _list = set(st.session_state[title_session_key]['b_select'])
            _list.add(select_title)
            st.session_state[title_session_key]['b_select'] = list(_list)
    
    options_b = st.multiselect(
        "选中的词条",
        list(st.session_state[title_session_key]['select_option']),
        default=st.session_state[title_session_key]['b_select'],
    )
    if len(options_b) != len(st.session_state[title_session_key]['b_select']):
        # print("modify",len(st.session_state[title_session_key]['b_select']), len(options_b))
        st.session_state[title_session_key]['b_select'] = options_b
      
    # st.divider()   
    # with st.container(border=True):
    #     custom_title = st.text_input("自定义文案主题", "summer", key="custom_title_b")
    #     with st.empty():
    #         custom_text = get_llm_article(custom_title)
    #         txt_b = st.text_area(
    #             f"自定义文案 - {custom_title}",
    #             custom_text,
    #             key="custom_area_b"
    #         )
    #     if custom_title != "":
    #         token_set_1, annotated_text_list = get_token(txt_b)
    #         token_data_list_2.append({
    #             'ID':0,
    #             'title':f"*{custom_title}*",
    #             'token_set':token_set_1,
    #             'plaintext': txt_b
    #         })
    
    # st.markdown("**token 展示区**")
    
    # if custom_title != "":
    #     with st.container(border=True):
    #         st.markdown(f"sentence: **{custom_title}**")
    #         annotated_text(annotated_text_list)


    # for title in options_b:
    #     # ID = st.session_state[title_session_key]['name_ID_dict'][title]
    #     # plaintext = get_plaintext(ID)
    #     # token_set_1, annotated_text_list = get_token(plaintext)
        
    #     # token_data_list_2.append({
    #     #     'ID':ID,
    #     #     'title':title,
    #     #     'token_set':token_set_1
    #     # })
        
    #     # st.text(f"sentence: {title}")
    #     # annotated_text(annotated_text_list)
    #     # st.divider()
        
    #     with st.container(border=True):
    #         ID = st.session_state[title_session_key]['name_ID_dict'][title]
    #         plaintext = get_plaintext(ID)
    #         token_set_1, annotated_text_list = get_token(plaintext)
    #         token_data_list_2.append({
    #             'ID':ID,
    #             'title':title,
    #             'token_set':token_set_1,
    #             'plaintext': plaintext
    #         })
    #         st.markdown(f"sentence: **{title}**")
    #         annotated_text(annotated_text_list)
    #     st.markdown("**token 展示区**")
    
    # if custom_title != "":
    #     with st.container(border=True):
    #         st.markdown(f"sentence: **{custom_title}**")
    #         annotated_text(annotated_text_list)

for title in options_b:
    ID = st.session_state[title_session_key]['name_ID_dict'][title]
    plaintext = get_plaintext(ID)
    token_set_1, annotated_text_list = get_token(plaintext)
    token_data_list_2.append({
        'ID':ID,
        'title':title,
        'token_set':token_set_1,
        'plaintext': plaintext
    })
    
with col1:
    
    st.markdown("**token 展示区**")
    ss = set(options_a)
    ss.update(options_b)
    for title in ss:
        with st.container(border=True):
            ID = st.session_state[title_session_key]['name_ID_dict'][title]
            plaintext = get_plaintext(ID)
            token_set_1, annotated_text_list = get_token(plaintext)
            token_data_list_1.append({
                'ID':ID,
                'title':title,
                'token_set':token_set_1,
                'plaintext': plaintext
            })
            st.markdown(f"sentence: **{title}**")
            annotated_text(annotated_text_list)

with col2:
    if token_data_list_1 and token_data_list_2:
        # 汇总表, 平均距离
        # 多对多,笛卡尔积
        
        token_data_list = []
        token_title_tuple_set = set()
        for item_a in token_data_list_1:
            for item_b in token_data_list_2:
                if item_a['title'] > item_b['title']:
                    token_title_tuple = (item_b['title'],item_a['title'])
                else:
                    token_title_tuple = (item_a['title'],item_b['title'])
                if token_title_tuple in token_title_tuple_set:
                    continue
                token_title_tuple_set.add(token_title_tuple)
                token_data_list.append((item_a, item_b))
                    
                
        
        container_summary = col2.container(border=True)

        summary_list = []
        
        st.markdown("**句子分区**, 两个词条间的 similarity")
        for item_a, item_b in token_data_list:
            df, df_out = calculate_cartesian_product_similarity_pgs(
                # tuple(int(item[0]) for item in token_set_1 - token_set_2), 
                # tuple( int(item[0]) for item in token_set_2 - token_set_1)
                tuple(item_a['token_set'] - item_b['token_set']), 
                tuple(item_b['token_set'] - item_a['token_set']),
                tuple(item_b['token_set'] & item_a['token_set']),
            )
            tree_similarity = get_tree_similarity(tuple(item[0] for item in item_a['token_set']),
                                tuple(item[0] for item in item_b['token_set']))
            # st.text(f"{item_a['title']} 和 {item_b['title']}")
            
            # print("tree_similarity",tree_similarity)
            if not len(df):
                continue
            with st.expander(f"{item_a['title']} 和 {item_b['title']}"):
                st.text("无效 similarity")
                st.dataframe(df_out)
                st.text("有效 similarity")
                st.dataframe(df)
                st.text(f"平均最大 similarity: {df['similarity'].mean().round(4)}")
                summary_list.append({
                    "token a": item_a['title'],
                    "token b": item_b['title'],
                    TITLE_SIMIARLITY_NAME: query_partner_distancles(item_a['ID'], item_b['ID']),
                    TOKEN_SIMIARLITY_NAME: df['similarity'].mean().round(4),
                    STS_SIMIARLITY_NAME: calculate_similarity(item_a['plaintext'], item_b['plaintext']),
                    TREE_SIMIARLITY_NAME: round(tree_similarity,4)
                })
                
                # st.divider()
                
        container_summary.markdown("**汇总区**, 词条的 similarity")
        df = pd.DataFrame(summary_list)
        config = {
            'source' : st.column_config.TextColumn('source',help="第一个title",),
            'dest' : st.column_config.TextColumn('dest', help="第二个节点",),
            TITLE_SIMIARLITY_NAME : st.column_config.NumberColumn(TITLE_SIMIARLITY_NAME,help="词条的 google similarity",),
            TOKEN_SIMIARLITY_NAME : st.column_config.NumberColumn('token similarity',help="词条文本首行的 tokenizer 的平均语义 similarity"),
            STS_SIMIARLITY_NAME : st.column_config.NumberColumn('STS similar',help="使用 jini embedding 模型计算的句子余弦距离, 越大越相似",),
        }
        container_summary.dataframe(df, column_config=config)
        
        data = []
        v1,_ = spearmanr(df[TOKEN_SIMIARLITY_NAME],df[STS_SIMIARLITY_NAME])
        va,_ = pearsonr(df[TOKEN_SIMIARLITY_NAME],df[STS_SIMIARLITY_NAME])
        data.append({"序列 a": TOKEN_SIMIARLITY_NAME, "序列 b": "STS similar", "spearmanr 相关性": v1, "pearsonr 相关性": va, })
        v2,_ = spearmanr(df[TITLE_SIMIARLITY_NAME],df[STS_SIMIARLITY_NAME])
        va,_ = pearsonr(df[TITLE_SIMIARLITY_NAME],df[STS_SIMIARLITY_NAME])
        data.append({"序列 a": TITLE_SIMIARLITY_NAME, "序列 b": "STS similar", "spearmanr 相关性": v2, "pearsonr 相关性": va})
        v3,_ = spearmanr(df[TITLE_SIMIARLITY_NAME],df[TOKEN_SIMIARLITY_NAME])
        va,_ = pearsonr(df[TITLE_SIMIARLITY_NAME],df[TOKEN_SIMIARLITY_NAME])
        data.append({"序列 a": TOKEN_SIMIARLITY_NAME, "序列 b": TITLE_SIMIARLITY_NAME, "spearmanr 相关性": v3, "pearsonr 相关性": va})
          
        container_summary.markdown("**相关性**")
        container_summary.dataframe(pd.DataFrame(data))
    