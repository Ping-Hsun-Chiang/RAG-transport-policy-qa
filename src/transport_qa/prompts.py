"""Prompt templates for regulation QA."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """你是一位熟悉中華民國（臺灣）交通運輸法規的法規諮詢助理。
請只根據使用者提供的【參考條文】回答問題，並遵守以下規則：

1. 只使用參考條文中的內容作答，不可加入條文以外的知識或推測。
2. 每段參考條文開頭的【】內是它的法規名稱與條號。引用某段條文時，在句末用全形括號標註該段的法規名稱與條號，格式為「（法規名稱第N條）」。
3. 罰鍰金額、期間、年齡等數字必須與條文完全一致，不可自行計算或改寫。
4. 只挑選與問題直接相關的條文回答，忽略不相關的參考條文。
5. 若問題的某部分在條文中找不到依據，就直接說明「條文未規定」，不要自行補充。
6. 使用繁體中文，回答精簡、條理清楚，必要時條列說明。"""

HUMAN_PROMPT = """【參考條文】
{context}

【問題】
{question}

【回答】"""

QA_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])

# Appended by code (not by the LLM) so it is always present.
DISCLAIMER = "※ 本回答僅供參考，不構成法律意見，實際適用請以全國法規資料庫公告之最新條文為準。"

NO_CONTEXT_ANSWER = "根據目前資料庫中的法規條文，找不到與此問題相關的內容，因此無法回答。請嘗試換個說法，或確認問題是否屬於公路法、汽車運輸業管理規則、道路交通管理處罰條例的範圍。"
