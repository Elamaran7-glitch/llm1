from fastapi import FastAPI, WebSocket
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
import uvicorn
from langchain_google_vertexai import VertexAI
import pandas as pd

app = FastAPI()

'''
q&a over data. gives text response.
'''

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'lumen-b-ctl-047-e2aeb24b0ea0.json'

model = VertexAI(model_name="gemini-1.5-pro-preview-0409", temperature=0, max_output_tokens=8192)
df = pd.read_csv('Jamaica_partial_cleaned.csv')

agent = create_pandas_dataframe_agent(
    model, df, verbose=True
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "exit":
            break
        response = agent.run(data)
        await websocket.send_json({"type": "text", "content": str(response)})

if __name__ == "__main__":
    uvicorn.run(app)