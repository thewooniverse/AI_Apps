# env variables and os.walk()
from dotenv import load_dotenv
import os
# Vector Support
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Model and chain
from langchain.chat_models import ChatOpenAI

# Text splitters
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

# The LangChain component we'll use to get the documents - way we'll retrieve vectors
from langchain.chains import RetrievalQA

# other modules
import datetime





# loading env variables, and creating the LLM
openai_api_key = os.getenv('OPENAI_API_KEY')
load_dotenv()
llm = ChatOpenAI(model_name='gpt-4', openai_api_key=openai_api_key)

# vector storage
embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=openai_api_key)

# loading the documentation - for now, this is the variable we manually change to query against different documentations
library_name = "nitter_scraper-master"
root_dir = f'documentations/{library_name}'
docs = []

# Go through each folder
for dirpath, dirnames, filenames in os.walk(root_dir):
    
    # Go through each file
    for file in filenames:
        try: 
            # Load up the file as a doc and split
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e: 
            pass


print (f"You have {len(docs)} documents\n")
print ("------ Start Document ------")
print (docs[0].page_content[:300])


# create our docsearch engine and build our retriever
docsearch = FAISS.from_documents(docs, embeddings)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


query = "How can I use Nitter to scrape Twitter profiles (# of follwers, # of accounts following, # of posts, account creation date, accountid) - please write me a sample code I can readily use based on documentations"
output = qa.run(query)
print(output)

# write output messages into output files;

results_path = f'results/{library_name}.txt'
with open(results_path, 'a') as f:
    f.write(f"\n\n----{datetime.datetime.now()}----\n")
    f.write(output)



