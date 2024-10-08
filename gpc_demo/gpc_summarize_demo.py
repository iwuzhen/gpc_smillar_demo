import streamlit as st
import pandas as pd
import pathlib
from scipy.stats import pearsonr, spearmanr
from gpc_demo.setting import (
    TITLE_SIMIARLITY_NAME,
    TOKEN_SIMIARLITY_NAME,
    STS_SIMIARLITY_NAME
)


st.header("GPC 句子之间的相关性总结",divider='rainbow')

st.markdown("#### 方法论\n\n"
            "1. 数据对象是 wikipedia 中的词条以及词条的第一段内容 \n\n"
            "2. 计算的指标有 3 个, 词条从名计算 google similarity, 第一段话计算 token similarity, 以及使用 jina-embeddings 计算句子的 embddding 向量, 然后使用*余弦相似度*计算 STS similarity \n\n"
            "3. 对数据集中的所有条目进行计算, 得到 similarity 向量 \n\n"
            "4. 对 similarity 向量进行相关性计算,使用 spearmanr pearsonr 的方法, 得出相关性分数\n\n"
            )

st.markdown("#### 数据集\n\n"
            "1. wikipedia subject 数据集, 包含主要学科和部分二级学科,共138个条目\n\n"
            "2. US president 数据集, 包含美国前总统,共45个条目\n\n"
            )


st.markdown("#### 附录 \n\n"
            "##### 1. 相关系数: Spearman vs Pearson  \n\n"
            "###### Spearman Correlation Coefficient（ Spearman 相关系数）\n\n"
            "> Wikipedia Definition: In statistics, Spearman’s rank correlation coefficient or Spearman’s ρ, named after Charles Spearman is a nonparametric measure of rank correlation (statistical dependence between the rankings of two variables). It assesses how well the relationship between two variables can be described using a monotonic function.\n\n"
            "统计学中，以 Charles Spearman 命名的 Spearman相关系数(Spearman ρ) 是排序相关性（ranked values)的非参数度量。使用单调函数描述两个变量之间的关系的程度。\n\n"
            "重要推论：Spearman ρ 相关性可以评估两个变量之间的单调关系——有序，它基于每个变量的排名值(ranked value)而不是原始数据(original value) \n\n"
            "###### Pearson Correlation Coefficient(皮尔逊相关系数)\n\n"
            "> Wikipedia Definition: In statistics, the Pearson correlation coefficient also referred to as Pearson’s r or the bivariate correlation is a statistic that measures the linear correlation between two variables X and Y. It has a value between +1 and −1. A value of +1 is a total positive linear correlation, 0 is no linear correlation, and −1 is a total negative linear correlation. \n\n"
            "皮尔逊相关系数(r)，是衡量两个变量X和Y之间线性相关性的统计量。它的值介于 +1 和 -1 之间。+1 完全正线性相关，0 是无线性相关，-1 是完全负线性相关。\n\n"
            "重要推论：皮尔逊相关性只能评估两个连续变量之间的线性相关（仅当一个变量的变化与另一个变量的比例变化相关时，关系才是线性的） \n\n"
            "##### 2. embedding 模型 与余弦相似度\n\n"
            "余弦相似度关注的是向量的方向而非大小，这与embedding模型的目标相符，即捕捉数据的语义信息而不是它们的大小或频率。因此其广泛用于自然语言处理任务\n\n"
            )
st.markdown("""\n\n
**相关性区间**: \n\n
0.00到0.19：几乎无相关性。\n\n
0.20到0.39：低度相关性。\n\n
0.40到0.59：中度相关性。\n\n
0.60到0.79：显著相关性。\n\n
0.80到0.99：高度相关性。\n\n
""")
st.divider()

@st.cache_data(max_entries=1_000)
def get_df(name):
    if name == "wikipedia subject":
        score_path =pathlib.Path(__file__).joinpath("../../model/score_subject.csv.gz").resolve()
    elif name == "US president":
        score_path =pathlib.Path(__file__).joinpath("../../model/score_president.csv.gz").resolve()
    elif name == "1M couple":
        score_path =pathlib.Path(__file__).joinpath("../../model/score_1m.csv.gz").resolve()
    df = pd.read_csv(score_path,index_col=False,compression="gzip")
    return df

@st.cache_data(max_entries=5)
def get_line_data(df):
    line_data_1 = []
    line_data_2 = []
    line_data_3 = []
    xais = []
    for i in range(100):
        xais.append(i/100)
        basic_df_tmp = df[df[TITLE_SIMIARLITY_NAME] >= i/100]
        if len(basic_df_tmp[TOKEN_SIMIARLITY_NAME]) < 100:
            continue
        
        v1,_ = spearmanr(basic_df_tmp[TOKEN_SIMIARLITY_NAME],basic_df_tmp[STS_SIMIARLITY_NAME])
        v2,_ = spearmanr(basic_df_tmp[TITLE_SIMIARLITY_NAME],basic_df_tmp[STS_SIMIARLITY_NAME])
        v3,_ = spearmanr(basic_df_tmp[TITLE_SIMIARLITY_NAME],basic_df_tmp[TOKEN_SIMIARLITY_NAME])
        line_data_1.append(v1)
        line_data_2.append(v2)
        line_data_3.append(v3)
    return xais,line_data_1,line_data_2,line_data_3
    
    

with st.container(border=True):
    option = st.selectbox(
        "选择数据集",
        ("wikipedia subject", "US president", "1M couple"))

    static_basic_df = get_df(option)

    data_range = st.slider("过滤器: range of google similarity",min_value =0.0,max_value = 1.0,value=(0.5,0.9),step=0.01)
    basic_df = static_basic_df[(static_basic_df[TITLE_SIMIARLITY_NAME] >= data_range[0]) & (static_basic_df[TITLE_SIMIARLITY_NAME] <= data_range[1])]
    st.text(f"数据条数: {len(basic_df)}")

data = []
if len(basic_df[TOKEN_SIMIARLITY_NAME]) > 5:
    v1,_ = spearmanr(basic_df[TOKEN_SIMIARLITY_NAME],basic_df[STS_SIMIARLITY_NAME])
    va,_ = pearsonr(basic_df[TOKEN_SIMIARLITY_NAME],basic_df[STS_SIMIARLITY_NAME])
    data.append({"序列 a": TOKEN_SIMIARLITY_NAME, "序列 b": STS_SIMIARLITY_NAME, "spearmanr 相关性": v1, "pearsonr 相关性": va, })
    v2,_ = spearmanr(basic_df[TITLE_SIMIARLITY_NAME],basic_df[STS_SIMIARLITY_NAME])
    va,_ = pearsonr(basic_df[TITLE_SIMIARLITY_NAME],basic_df[STS_SIMIARLITY_NAME])
    data.append({"序列 a": TITLE_SIMIARLITY_NAME, "序列 b": STS_SIMIARLITY_NAME, "spearmanr 相关性": v2, "pearsonr 相关性": va})
    v3,_ = spearmanr(basic_df[TITLE_SIMIARLITY_NAME],basic_df[TOKEN_SIMIARLITY_NAME])
    va,_ = pearsonr(basic_df[TITLE_SIMIARLITY_NAME],basic_df[TOKEN_SIMIARLITY_NAME])
    data.append({"序列 a": TOKEN_SIMIARLITY_NAME, "序列 b": TITLE_SIMIARLITY_NAME, "spearmanr 相关性": v3, "pearsonr 相关性": va})
          
with st.container(border=True):
    xais,line_data_1,line_data_2,line_data_3 = get_line_data(static_basic_df)

    st.markdown("## 总体相似度相关性变化曲线 \n\n 从 x ~ 1 范围内的数据总体相关性")
    chart_data = pd.DataFrame({
       f"token-STS": line_data_1,
        f"google-STS": line_data_2,
       f"google-token": line_data_3,
       })

    # 设置 x 轴的值
    st.line_chart(chart_data)


    st.markdown("## 总体相似度相关性 \n\n 对全样本进行相关性计算")
    st.dataframe(pd.DataFrame(data))

data_1 = []
data_2 = []
data_3 = []
for key in basic_df['source'].drop_duplicates():
    df_tmp = basic_df[basic_df['source'] == key]
    a = df_tmp[STS_SIMIARLITY_NAME]
    b = df_tmp[TOKEN_SIMIARLITY_NAME]
    if len(a) < 5:
        continue
    v1,_ = spearmanr(a,b) 
    v2,_ = pearsonr(a,b) 
    doc = {
        'token': key,
        '序列 a': STS_SIMIARLITY_NAME,
        '序列 b': TOKEN_SIMIARLITY_NAME,
        'spearmanr 相关性': v1,
        'pearsonr 相关性': v2,
    }
    data_1.append(doc)

    a = df_tmp[TITLE_SIMIARLITY_NAME]
    b = df_tmp[TOKEN_SIMIARLITY_NAME]
    v1,_ = spearmanr(a,b) 
    v2,_ = pearsonr(a,b) 
    doc = {
        'token': key,
        '序列 a': TOKEN_SIMIARLITY_NAME,
        '序列 b': TITLE_SIMIARLITY_NAME,
        'spearmanr 相关性': v1,
        'pearsonr 相关性': v2,
    }
    data_2.append(doc)

    a = df_tmp[STS_SIMIARLITY_NAME]
    b = df_tmp[TITLE_SIMIARLITY_NAME]
    v1,_ = spearmanr(a,b) 
    v2,_ = pearsonr(a,b) 
    doc = {
        'token': key,
        '序列 a': STS_SIMIARLITY_NAME,
        '序列 b': TITLE_SIMIARLITY_NAME,
        'spearmanr 相关性': v1,
        'pearsonr 相关性': v2,
    }
    data_3.append(doc)
    

with st.container(border=True):
    st.markdown("## 分词条相似性 \n\n 按词条分组, 对分组内的相似度进行相关性计算")
    
    option = st.selectbox(
        "选择比较对象",
        (
         f"{STS_SIMIARLITY_NAME}-{TOKEN_SIMIARLITY_NAME}",
        f"{STS_SIMIARLITY_NAME}-{TITLE_SIMIARLITY_NAME}", 
         f"{TOKEN_SIMIARLITY_NAME}-{TITLE_SIMIARLITY_NAME}",))
    if option == f"{STS_SIMIARLITY_NAME}-{TITLE_SIMIARLITY_NAME}":
        df = pd.DataFrame(data_3)
    elif option == f"{TOKEN_SIMIARLITY_NAME}-{TITLE_SIMIARLITY_NAME}":
        df = pd.DataFrame(data_2)
    else:
        df = pd.DataFrame(data_1)
    st.dataframe(df)


config = {
    'source' : st.column_config.TextColumn('token a',help="第一个title",),
    'dest' : st.column_config.TextColumn('token b', help="第二个节点",),
    TITLE_SIMIARLITY_NAME : st.column_config.NumberColumn(TITLE_SIMIARLITY_NAME,help="词条的语义距离",),
    TOKEN_SIMIARLITY_NAME : st.column_config.NumberColumn(TOKEN_SIMIARLITY_NAME,help="词条文本首行的 tokenizer 的平均语义距离"),
    STS_SIMIARLITY_NAME : st.column_config.NumberColumn(STS_SIMIARLITY_NAME,help="使用 jini embedding 模型计算的句子余弦距离, 越大越相似",),
}

with st.container(border=True):
    st.markdown("## 词条相似度 (原始数据)")
    st.dataframe(basic_df, column_config=config)