from dataclasses import asdict, dataclass, fields

from dotenv import load_dotenv
from langchain_cohere.llms import Cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from traceloop.sdk import Traceloop
from traceloop.sdk.instruments import Instruments


@dataclass
class PromptVariables:
    number_of_years: int
    max_grade: str
    location: str
    route_grade: str


if __name__ == "__main__":
    load_dotenv()
    Traceloop.init(
        app_name="route_suggestions",
        disable_batch=True,
        instruments={Instruments.LANGCHAIN, },
    )
    input_variables = PromptVariables(
        20,
        "7b",
        "the Swiss canton of Glarus",
        "6c",
    )
    llm = Cohere(temperature=0)
    prompt_template = PromptTemplate(
        input_variables=[fields(input_variables)],
        template=(
            "I have been mountaineering and climbing for over {number_of_years} years,"
            " in climbing up to a grade of {max_grade}. Suggest me established routes"
            " in {location} up to a climbing grade of {route_grade}. Your main source"
            " of information are the following webpages: thecrag.com and hikr.org."
        ),
    )
    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke(asdict(input_variables))
    print(response)
