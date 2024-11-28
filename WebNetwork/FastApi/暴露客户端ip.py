from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()


@app.post("/ip")
async def report_ip(request: Request):
    client_ip = request.client.host
    print(f"Client IP: {client_ip}")
    return {"message": f"your ip: {client_ip}"}


# Run the server with `uvicorn main:app --host 0.0.0.0 --port 8000`
# client use: curl -X POST http://your_server_ip:you_port/ip
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6666)
