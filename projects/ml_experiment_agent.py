from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal
from pydantic import BaseModel, Field

# Schema for metric_threshold

class MetricThreshold(BaseModel):
    metric: Literal['accuracy', 'f1', 'loss'] = Field(description='The selected metric')
    threshold: float = Field(description='Value of the metric to compare. Accuracy and f1 should be between 0 and 1')
    comparison: Literal['gt', 'gte', 'lt', 'lte'] = Field(description='The comparison operator. gt is greater than, gte is greater than or equal to, lt is less than, lte is less than or equal to')


@tool
def search_runs(dataset: Annotated[Literal['imdb', 'cifar10', 'mnist', 'fraud-tx'] | None, 'The dataset to filter'] = None,
                model: Annotated[Literal['resnet18', 'resnet50', 'vit-small', 'logreg'] | None, 'The model to filter'] = None, 
                metric_thresholds: Annotated[list[MetricThreshold] | None, 'All threshold conditions must pass'] = None ):
    '''
    find runs by dataset, model, metric threshold
    '''

    # Dataset filter
    if dataset:
        pass

    return 'Hello'

print(search_runs.tool_call_schema.model_json_schema())