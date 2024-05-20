import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.base import RunnableSequence
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class MovieRecommender:
    def __init__(self):
        # Load the environment variables from the .env file
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize the embedding model
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Load data into LangChain
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir,"imdb.csv")
        loader = CSVLoader(file_path=file_path)
        data = loader.load()

        # Split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunked_documents = text_splitter.split_documents(data)

        # Create embedder
        store = LocalFileStore("./cache/")
        underlying_embeddings = OpenAIEmbeddings()
        self.embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )

        # Create vector store using FAISS
        self.vector_store = FAISS.from_documents(chunked_documents, self.embedder)
        self.vector_store.save_local("vector_store")

        # Create the components
        self.prompt_template = ChatPromptTemplate.from_template(
            "{user_input}"
        )
        self.retriever = self.vector_store.as_retriever()
        self.chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.parser = StrOutputParser()

        # Create the sequence
        self.runnable_chain = (
            self.prompt_template
            | self.chat_model
            | self.parser
        )
    async def provide_rec(self, user_input):
      return self.runnable_chain.astream({"user_input": "bats"})


    def recommend(self, user_input):
        output_chunks = self.runnable_chain.invoke({"user_input": user_input})
        return ''.join(output_chunks)