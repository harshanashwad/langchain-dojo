'''
You are building an automated email triage system for a customer support team. Your task is to implement a function that classifies incoming customer 
emails and routes them to the appropriate department handler.

Implement the classify_and_route_email function
'''

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI

class EmailClassification(BaseModel):
    category: Literal["Technical Support", "Billing", "General Feedback"] = Field(
        description="The category of the customer email"
    )
    confidence: float = Field(description="Confidence score between 0 and 1")
    routed_response: str = Field(
        description="The response from the specialized department"
    )

def invoke_llm(prompt: str):
    llm = ChatOpenAI(model='gpt-4o-mini')
    response = llm.invoke(prompt)
    return response.content

def classify_and_route_email(email_content: str, mock_llm_func) -> EmailClassification:
    
    # Step 1: Classification
    
    prompt_template = PromptTemplate.from_template('You are an customer support email classifier. Classify the email delimited by triple backticks into one of Technical Support, Billing or General Feedback. And assign a confidence score (0 to 1) for how confident you are in your classification. The routed_response can be empty for this request \n```{email_content}```\n{format_instructions}')
    parser = PydanticOutputParser[EmailClassification](pydantic_object=EmailClassification)
    classify_prompt = prompt_template.format(email_content=email_content, format_instructions=parser.get_format_instructions())

    response = mock_llm_func(classify_prompt)
    classification_response = parser.parse(response)

    print(classification_response)

    # Step 2: Routing

    # Define prompts for departments
    techinical_prompt_template = PromptTemplate.from_template('You are a Technical Support Specialist. Analyze the email content delimited by triple backticks and send an appropriate response\n {email_content}')
    billing_prompt_template = PromptTemplate.from_template('You are a Billing Issues Specialist. Analyze the email content delimited by triple backticks and send an appropriate response\n {email_content}')
    general_prompt_template = PromptTemplate.from_template('You are a General Feedback Specialist. Analyze the email content delimited by triple backticks and send an appropriate response\n {email_content}')

    # Use RunnableBranch to pick route the request based on category
    chain = RunnableBranch(
        (lambda x: classification_response.category == 'Billing', billing_prompt_template),
        (lambda x: classification_response.category == 'Technical Support', techinical_prompt_template),
        general_prompt_template
    ) | RunnableLambda(mock_llm_func) 

    routed_response = chain.invoke({
        'email_content': email_content
    })

    classification_response.routed_response = routed_response
    return classification_response

final_response =  classify_and_route_email('My iphone is not working. Keeps rebooting several times a day!!', mock_llm_func=invoke_llm)
print(final_response)
