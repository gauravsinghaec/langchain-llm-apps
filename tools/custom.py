from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field


def get_character_count(input_str: str):
    return len(input_str)


class GetCharacterCountInput(BaseModel):
    """
    Pydantic arguments schema for get_character_count method
    """

    input_str: str = Field(..., description="The text to be counted for")


char_count_tool = StructuredTool.from_function(
    func=get_character_count,
    args_schema=GetCharacterCountInput,
    description="Function to count the number of charaters in a text.",
)
