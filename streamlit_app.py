# streamlit run streamlit_app.py
import streamlit as st
import json

from src.vectorrag.config import MyConfig as mcfg

# DB 연결
from src.vectorrag.db import loading_vectorstore
from src.graphrag.connect_neo4j import connect_n4j_graph

# import for Plain LLM
from langchain_ollama import OllamaLLM

# import for Vector RAG
from src.vectorrag.query_answer import query_answering as vectorrag_query_answering

# import for Graph RAG
from src.graphrag.retrieve import graph_retriever, vector_retriever
from src.graphrag.query_answer import query_answering as graphrag_query_answering

# import for Hybrid RAG
from src.graphrag.query_answer import query_answering_hybrid as graphrag_query_answering_hybrid
 
st.set_page_config(
    page_title="LLM-Based RAG Comparison Framework",
    layout="wide",
)
 
# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG 입력값 설정")
    
    # 미리 선정한 질문 목록
    preset_questions = [
        "직접 입력",
        "에픽론에 대해 설명해줘",
        "업계에서 부정적인 평가를 받고 있는 회사와 그 이유는?",
        "델타테크 창업자가 AlphaAI 퇴사 후 설립한 회사에 투자한 기관은?"
    ]
    
    selected = st.selectbox("질문 선택", preset_questions)
    
    if selected == "직접 입력":
        query = st.text_area(
            "질문 입력",
            value="",
            height=100,
        )
    else:
        query = st.text_area(
            "질문 입력",
            value=selected,
            height=100,
        )
    
    top_k = st.number_input("검색 결과 수 (k)", min_value=1, max_value=10, value=3)
    run = st.button("▶ 실행", use_container_width=True, type="primary")
 
    st.divider()
    
    st.caption("DB 연결")
    vectorstore = None
    graph = None
    
    # vector DB 연결 확인
    try:
        vectorstore = loading_vectorstore()
        if vectorstore:
            st.success("VectorStore 연결됨")
        else:
            st.error(f"VectorStore 오류")
    except Exception as e:
        st.error(f"VectorStore 오류: {e}")
    
    # graph DB 연결 확인
    try:
        graph = connect_n4j_graph()
        if graph:
            st.success("Graph DB 연결됨")
        else:
            st.error(f"Graph DB 오류")
    except Exception as e:
        st.error(f"Graph DB 오류: {e}")
 
# ── 메인 ──────────────────────────────────────────────────────────────────────
st.title("LLM-Based RAG Comparison Framework")
st.caption("동일 질문에 대해 각 RAG 방식의 검색 문서와 답변을 나란히 비교합니다.")
 
if not run:
    st.info("사이드바에서 질문을 입력하고 **실행** 버튼을 누르세요.")
    st.stop()
 
st.markdown(f"> **질문:** {query}")
st.divider()
 
# ── 실행 ──────────────────────────────────────────────────────────────────────
# col_llm, col_divider_1, col_vec = st.columns(3, gap="medium")
col_llm, col_divider_1, col_vec = st.columns([10, 1, 10], gap="small")
# ────────────────────────────────────────────────
# 1: LLM (no RAG)
# ────────────────────────────────────────────────
with col_llm:
    st.subheader("🤖 LLM (no RAG)")
    st.caption("내부 지식(정보) 기반 추론")
    with st.spinner("LLM 추론 중..."):
        try:
            no_rag_llm = OllamaLLM(model=mcfg.ollama_model)
            llm_query = query + "\n 한 문단 정도로 간단하게 답변하라."
            no_rag_respose = no_rag_llm.invoke(llm_query)
            st.info(no_rag_respose)
        except Exception as e:
            st.error(f"LLM 오류: {e}")
            
# ────────────────────────────────────────────────
with col_divider_1:
    st.markdown(
        """
        <div style="
            border-left: 1px solid rgba(255,255,255,0.15);
            height: 100%;
            min-height: 600px;
            margin: 0 auto;
            width: 1px;
        "></div>
        """,
        unsafe_allow_html=True
    )
    
# ────────────────────────────────────────────────
# 2: Vector RAG
# ────────────────────────────────────────────────
with col_vec:
    st.subheader("📐 Vector RAG")
    st.caption("벡터 유사도 기반 검색")
 
    with st.spinner("Basic RAG 실행 중..."):
        try:
            vectorstore = loading_vectorstore()
            vectorrag_response = vectorrag_query_answering(query, vectorstore)
            docs_with_score = vectorstore.similarity_search_with_score(query, k=int(top_k))
 
            st.markdown("**관련 문서**")
            for i, doc in enumerate(docs_with_score):
                with st.expander(f"문서 {i+1}    Score : {doc[1]:.4f}", expanded=False):
                    st.write(doc[0].page_content)
 
            st.markdown("**답변**")
            st.info(vectorrag_response.content)
 
        except Exception as e:
            st.error(f"Basic RAG 오류: {e}")

st.divider()
# col_graph, col_divider_2, col_hybrid = st.columns(3, gap="medium")
col_graph, col_divider_2, col_hybrid = st.columns([10, 1, 10], gap="small")
# ────────────────────────────────────────────────
# 3: Graph RAG
# ────────────────────────────────────────────────
with col_graph:
    st.subheader("🕸️ Graph RAG")
    st.caption("지식 그래프 검색")
 
    with st.spinner("Graph RAG 실행 중..."):
        try:
            graphrag_response = graphrag_query_answering(query)
            graph_docs, entities = graph_retriever(query)
            
            fixed = "[" + graph_docs.replace("{'node", ", {'node")[1:] + "]"
            fixed = fixed.replace("'", '"')
            result = json.loads(fixed)
#             vector_docs = vector_retriever(query)
            
            st.markdown(f"**{', '.join(entities.entities)}에 대한 그래프 검색 결과 (5개 까지)**")
            if graph_docs:
                for i, doc in enumerate(result[:5]):
                    with st.expander(f"subgraph {i+1}", expanded=False):   # Score : {doc[1]:.4f}", expanded=True):
                        st.write(doc)

            else:
                st.caption("그래프 결과 없음")
 
            st.markdown("**답변**")
            st.info(graphrag_response.content)
 
        except Exception as e:
            st.error(f"Graph RAG 오류: {e}")
            
# ────────────────────────────────────────────────
with col_divider_2:
    st.markdown(
        """
        <div style="
            border-left: 1px solid rgba(255,255,255,0.15);
            height: 100%;
            min-height: 600px;
            margin: 0 auto;
            width: 1px;
        "></div>
        """,
        unsafe_allow_html=True
    )
    
# ────────────────────────────────────────────────
# 4: Graph RAG + Vector 통합 결과
# ────────────────────────────────────────────────
with col_hybrid:
    st.subheader("🧩 Graph RAG + Vector Hybrid")
    st.caption("Graph 검색과 Vector 검색을 합쳐 최종 답변을 생성")
 
    with st.spinner("Hybrid RAG 실행 중..."):
        try:
            graph_docs_hybrid, entities= graph_retriever(query)
            vector_docs = vector_retriever(query,k=int(top_k))
            graphrag_response_hybrid = graphrag_query_answering_hybrid(query)
            
            fixed_hybrid = "[" + graph_docs_hybrid.replace("{'node", ", {'node")[1:] + "]"
            fixed_hybrid = fixed_hybrid.replace("'", '"')
            result_hybrid = json.loads(fixed_hybrid)
            
            st.markdown(f"**{', '.join(entities.entities)}에 대한 그래프 검색 결과 (5개 까지)**")
            if graph_docs:
                for i, doc in enumerate(result[:5]):
                    with st.expander(f"subgraph {i+1}", expanded=False):   # Score : {doc[1]:.4f}", expanded=True):
                        st.write(doc)

            else:
                st.caption("그래프 결과 없음")

            st.markdown(f"**관련 문서**")
            for i, doc in enumerate(vector_docs):
                    with st.expander(f"문서 {i+1}", expanded=False):
                        st.write(doc)
            st.markdown("**답변**")
            st.info(graphrag_response_hybrid.content)
    
        except Exception as e:
            st.error(f"통합 결과 오류: {e}")