import os
import asyncio
from kaggle_secrets import UserSecretsClient
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from state import APP_NAME, USER_ID, SESSION

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error "
    )

db_url = "sqlite:///my_agent_data.db"  # Local SQLite file
session_service = DatabaseSessionService(db_url=db_url)

from root_system_agent import root_agent

async def run_session(
    runner_instance: Runner, user_queries: list[str] | str, session_id: str = "default"
):
    """Helper function to run queries in a session and display responses."""
    print(f"\n### Session: {session_id}")

    try:
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )
    except:
        session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=session_id
        )

    # Converting single query -> list
    if isinstance(user_queries, str):
        user_queries = [user_queries]

    # Here we process each query
    for query in user_queries:
        print(f"\nUser > {query}")
        query_content = types.Content(role="user", parts=[types.Part(text=query)])

        #  agent response
        async for event in runner_instance.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"Model: > {text}")

runner = Runner(
    agent=root_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

print("✅ Runner instance defined.")
print("✅ MRG control tools created")
print("completed trading layer")

trade_query = "The optimal trade is to BUY 100 shares of MSFT at the current market price."

asyncio.run(run_session(
    runner_instance=runner, 
    user_queries=trade_query, 
    session_id="SESSION_A"
))

print("✅ ADK components imported successfully.")
