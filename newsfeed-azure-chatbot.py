import streamlit as st
import os
import json
from scipy import spatial
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from newsapi import NewsApiClient
from newspaper import Article
from newspaper import Config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import AzureOpenAI,AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load Env Variables
load_dotenv()

## News API
newsapi = NewsApiClient(api_key=os.getenv('news_api_key'))

## Azure
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["OPENAI_API_TYPE"] = os.getenv('OPENAI_API_TYPE')
os.environ["OPENAI_API_VERSION"] = os.getenv('OPENAI_API_VERSION')

# Set up JSON Formatting
class QuerySchema(BaseModel):
    expanded_queries: list[str] = Field(description="the list of TOP 10 queries")

parser = JsonOutputParser(pydantic_object=QuerySchema)
format_instructions = parser.get_format_instructions()

# Date variables
today = datetime.today()
prev_day = (today - timedelta(days=0)).strftime('%Y-%m-%d')
prev_30day = (today - relativedelta(months=+1)).strftime('%Y-%m-%d')

# Helper Functions
def cosine_similarity(x,y):
    return 1 - spatial.distance.cosine(x, y)

def get_embeddings(text):
    embeddings = AzureOpenAIEmbeddings(model=EMBEDDING)
    response = embeddings.embed_query(text)
    return response

# Get Response from News API
def search_news(query, num_articles=5, from_datetime = prev_30day,to_datetime = prev_day):
    
    response = newsapi.get_everything(q=query,
                                      from_param=from_datetime,
                                      to=to_datetime,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=num_articles
                                      )


    return response

# Get Content From URL
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

def get_content_from_url(url, text_max_length=2000):
    try:
        page = Article(url)
        page.download()
        page.parse()
        
        return page.text[:text_max_length].replace('$','\\$') # Avoid unnecessary markdown display for $
    
    except:
        return 'Error'


# Main Functions

def decide_chain(user_query, chat_history):
    decide_system_prompt = """You are an assistant good at finding recent news articles on a given topic using a NEWS API.\
           You have 2 tools, 'RAG' and 'Direct Answer'. Given a chat history and the latest user question, you need to decide which tool to use to respond.\
           You will use the tool 'RAG' if the question is about news topic and you want to look up more information.\
           You will use the tool 'Direct Answer' if the question is generic or does not pertain to any potential news topics.\
           Question: {question}
           Which tool do you want to use? Only reply 'RAG' or 'Direct Answer'."""

    decide_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", decide_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    llm = AzureChatOpenAI(deployment_name = "gpt-4o", model='gpt-4o')

    decide_chain = decide_prompt | llm | StrOutputParser()

    return decide_chain.invoke({
        "question": user_query,
        "chat_history": chat_history
        
    })

def genral_answer(user_query, chat_history):
    general_system_prompt =  """You are an assistant good at finding recent news articles on a given topic using a NEWS API.\
                            Please Respond to the following question:
                            Question: {question}
                            Answer:"""

    general_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", general_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    llm = AzureChatOpenAI(deployment_name = "gpt-4o", model='gpt-4o')
    general_chain = general_prompt | llm

    return general_chain.stream({
        "question": user_query,
        "chat_history": chat_history
        
    })



def get_relevant_queries(user_query):

    template = """
                You will be generating search queries to find recent news articles on a given topic using a NEWS API.\

                The topic is: {user_query}\

                Your goal is to generate many potential search queries that are relevant to the topic and valid in year 2025. To do this:\
                - Use different keywords and phrases related to the topic \
                - Vary the specificity of your queries, making some more narrow and others more broad\
                - Be creative and come up with as many distinct query ideas as you can\

                First, brainstorm at least 20 distinct query ideas of varying specificity. Please don't return the output of brainstorming.\

                Then select the 10 best, most promising queries. Provide them in the following format: {format_instructions}
                Don't include '```' in the answer.
                
                """

    prompt = PromptTemplate.from_template(template,partial_variables={"format_instructions": parser.get_format_instructions()})
    
    llm = AzureChatOpenAI(deployment_name = "gpt-4o", model='gpt-4o')
        
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "user_query": user_query,
    })

def generate_hypothetical_answer(user_query):
    template = """
                Here is a question from a user:{user_query}

                Please make up a hypothetical answer to this question. Imagine you have all the details needed to answer it, even if those details are not real. 
                Do not use actual facts in your answer. Instead, use placeholders like 'EVENT affected something' or 'NAME mentioned something on DATE' to represent key details.
                Limit your answer in 500 characters.

                
         """

    prompt = PromptTemplate.from_template(template)
    llm = AzureChatOpenAI(deployment_name = MODEL, model_name = MODEL)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "user_query": user_query,
    })

def pick_news(queries,hypothetical_answer_embedding):
    articles = []
    for query in queries:
        result = search_news(query)
        if result['status'] == 'ok':
            articles = articles + result['articles']
        else:
            raise Exception(result["message"])
            
    #Remove duplicates
    articles = {article["url"]: article for article in articles}
    
    # Get Content From url
    for key in articles.keys():
        content_from_url = get_content_from_url(key)
        if content_from_url!='Error':
            articles[key]['content'] = content_from_url
            
    articles = list(articles.values())
    
    
    articles_prepare_embedd =  [
        f"{article['title']} {article['content'][0:500]}"
        for article in articles
    ]
    
    article_embeddings =  [get_embeddings(article) for article in articles_prepare_embedd]
    
    cosine_similarities = []
    for article_embedding in article_embeddings:
        cosine_similarities.append(cosine_similarity(hypothetical_answer_embedding, article_embedding))
        
    scored_articles = zip(articles, cosine_similarities)
    sorted_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)


    formatted_top_results = [
        {
            "Title": article["title"],
            "Url": article["url"],
            "Content": article['content']
        }
        for article, _score in sorted_articles[0:10]
    ]
    
    return formatted_top_results

def summarize_top_reuslts(formatted_top_results,user_query):
    template = """
                Generate an answer to the user's question based on the given search results.
                TOP_RESULTS: {formatted_top_results}
                USER_QUESTION: {user_query}

                Please leverage all the TOP_RESULTS to answer USER_QUESTION and include as many details as possible in the answer.
                Please also replace the dollar sign($) as '\\$' in the answer for better markdown display.
                Reference the relevant search result urls as markdown links.
                

                
         """

    prompt = PromptTemplate.from_template(template)
    llm = AzureChatOpenAI(deployment_name = MODEL, model_name = MODEL)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "user_query": user_query,
        "formatted_top_results": formatted_top_results
    })



# Set Streamlit page configuration
st.set_page_config(page_title="🗞️News Feed Assistant🤖", layout="centered")

## Hide header
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🗞️News Feed Assistant🤖")
st.markdown(
    """ 
        > :rainbow[**A Chatbot for News**, powered by -  [LangChain](https://python.langchain.com/v0.2/docs/introduction/) + 
        [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry) + 
        [OpenAI](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4) + 
        [Streamlit](https://streamlit.io)]
        """
)

st.sidebar.warning(
        "Please be advised that this chatbot is intended for general informational purposes only. \
         \n\nTo ensure your privacy and security, do not input any internal, sensitive, or confidential information into this chatbot. \
        "
    ,icon="⚠️")


# Set up sidebar with various options
with st.sidebar.expander(" 🛠️ Settings ", expanded=False):

    MODEL = st.selectbox(
        label="LLM_Model",
        options=[
            "gpt-4o",
            "gpt-5-chat",
            "gpt-3.5-turbo"
        ],
    )
    
    EMBEDDING = st.selectbox(
        label="Embedding_Model",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
    )
    
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your assistant, here to help you find news that interests you. How may I assist you today?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI",avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human', avatar="🧐"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
download_list = []
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    download_list.append('**User Question**: '+ '**' + user_query + '**')

    with st.chat_message('Human', avatar="🧐"):
        st.markdown(user_query)
    
    tool = decide_chain(user_query, st.session_state.chat_history)
    print(tool)

    if tool != 'RAG':
        with st.chat_message("AI",avatar="🤖"):
            response = st.write_stream(genral_answer(user_query, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(content=response))

    else:
        # Generate Related Queries
        st.markdown("**Relevant Queries:**")
        with st.chat_message("AI",avatar="🤖"):
            response = st.write_stream(get_relevant_queries(user_query))

        st.session_state.chat_history.append(AIMessage(content=response))

        queries = json.loads(response)['expanded_queries']
        queries.insert(0,user_query)

        # Generate Hypothetical Answer
        st.markdown("**Hypo Answers:**")
        with st.chat_message("AI", avatar="🤖"):
            response = st.write_stream(generate_hypothetical_answer(user_query))

        st.session_state.chat_history.append(AIMessage(content=response))
        hypothetical_answer_embedding = get_embeddings(response)

        # Retrieving TOP Articles
        with st.spinner("Retrieving Articles..."):
            formatted_top_results = pick_news(queries,hypothetical_answer_embedding)
            formatted_top_results_display = []
            for result in formatted_top_results:
                    formatted_top_results_display.append('\n\n'.join(["\n\n".join([f'**{key}**',value]) for key,value in result.items()]))
            formatted_top_results_display = f"\n\n{'-'*50}\n\n".join(formatted_top_results_display)

        st.markdown("**TOP Articles:**")
        with st.chat_message("AI", avatar="🤖"):
            st.write(formatted_top_results_display)

        # Summarized Output
        st.markdown("**Summarized Output:**")
        with st.chat_message("AI", avatar="🤖"):
            response = st.write_stream(summarize_top_reuslts(formatted_top_results,user_query))

        st.session_state.chat_history.append(AIMessage(content=response))

        download_list.append('**News Feed Assistant**: ' + response)
        download_txt = '\n\n'.join(download_list)
        
        if download_list:
            st.download_button('Download',download_txt,file_name=f"{user_query}-{datetime.now().strftime('%Y%m%d%H%M%S')}.md")

