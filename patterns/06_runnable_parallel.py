"""
PATTERN: RunnableParallel
DAY: 6

PURPOSE: RunnableParallel executes multiple operations (can be functions, Runnables) at the same time

USE WHEN: You want to branch from a certain point in a chain and do multiples operations at the same time. For example, running multiple chains together!

FLOW: 


<input_dict>  → RunnableParallel(
    key=runnable,
    key=callable,
    key=dict
    )
  → <output_dict>

Each value is a branch, each key becomes a key in the output dict.


NEW CONSTRUCTS:
RunnableParallel: Takes in a list of Runnables (RunnableLambda for example), callable or a dict and executes them simultaneously
"""

resume = """
John Doe | Software Engineer
Skills: React, Figma, Canva, Good UI/UX principles
Experience: 3 years at a fintech startup building backend services.
Reduced API latency by 40% through query optimization.
Communication: Writes clear technical docs, led weekly team standups.
"""

job_desc = """
We are looking for a Backend Engineer with strong Python skills,
experience with REST APIs and databases, and the ability to communicate
clearly with cross-functional teams. FastAPI experience is a plus.
"""

# skills: "Rate the skills match from 1-10 for this resume: {resume} against this job: {job_desc}. Return a number only."
# experience: "Rate the experience match from 1-10 for this resume: {resume} against this job: {job_desc}. Return a number only."
# tone: "Rate the communication quality from 1-10 based on this resume: {resume}. Return a number only."


from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
parser = StrOutputParser()

skills_prompt = ChatPromptTemplate.from_messages(
    messages=[
        ('user', 'Rate the skills match from 1-10 for this resume: {resume} against this job: {job_desc}. Return a number only.')
    ]
)

experience_prompt = ChatPromptTemplate.from_messages(
    messages=[
        ('user', 'Rate the experience match from 1-10 for this resume: {resume} against this job: {job_desc}. Return a number only."')
    ]
)

communication_prompt = ChatPromptTemplate.from_messages(
    messages= [
        ('user', "Rate the communication quality from 1-10 based on this resume: {resume}. Return a number only.")
    ]
)

runnable_parallel = RunnableParallel(
    skills_score=skills_prompt | llm | parser,
    experience_score=experience_prompt | llm | parser,
    communication_score=communication_prompt | llm | parser
)

response = runnable_parallel.invoke({
    'resume': resume,
    'job_desc': job_desc
})

print(response)

'''
POST-ANALYSIS BLOCK

RunnableParallel: input_dict goes in, output_dict comes out. Each keyword can be a runnable (say a function wrapped using RunnableLambda), callable (like a chain) or a dict (short hand notation for another runnable parallel as a branch)

Use it when you want to branch out doing different operations and getting the results. We can fan out, fan out further, accumulate results and send
to next step. But we need to ensure, next step is getting the dict with all expected keys in each step in your chain as an architect

One-liner:
RunnableParallel fans out one input to multiple chains simultaneously and collects results into a dict for the next step.
'''