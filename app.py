import chainlit as cl

from notebooks.imdb_file import MovieRecommender

recommender = MovieRecommender()

@cl.step
def tool():
    return "Response from the tool!"

async def build_query(message: cl.Message):
    return f"{message.content}"

async def run_and_analyze(parent_id: str, query: str):
    return f"{query}"

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    # Call the tool
    tool()
    
    recommendation = recommender.recommend(message.content)
    await cl.Message(content=recommendation).send()

    # recommendation_stream = await recommender.provide_rec(message.content)

    # async for chunk in recommendation_stream:
    # # Send the final answer.
    #     await cl.Message(content=chunk).send()

