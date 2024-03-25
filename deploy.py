# Import FastAPI
from fastapi import FastAPI
from Jeet_copy import load_and_predict
# Create an instance of the FastAPI class
app = FastAPI()

# Define a path operation for the root ("/") with a GET operation
@app.get("/app")
async def read_root():
    return {"message": "Hello, World!"}

# Define an additional path operation for "/greet/{name}" with a GET operation
@app.get("/predict/{query}")
async def predict_query(query: str):
    message = None
    predictions = load_and_predict(input_features = query)
    if predictions == 1:
        message = 'Harmful ⚠️'
    else:
        message = 'Not harmful 😊'
    response  = "Your SQL Query is " + f"{message}"   
    return response


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(deploy_test, host='0.0.0.0', port=8000)