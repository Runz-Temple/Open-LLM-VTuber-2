import os
import re
import shutil
import atexit
import json
import asyncio
import subprocess
from typing import List, Dict, Any
import yaml
import numpy as np
import chardet
from loguru import logger
from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
import uvicorn
from main import OpenLLMVTuberMain
from live2d_model import Live2dModel
from tts.stream_audio import AudioPayloadPreparer
import __init__

class WebSocketServer:
    def __init__(self, open_llm_vtuber_main_config: Dict | None = None):
        logger.info(f"t41372/Open-LLM-VTuber, version {__init__.__version__}")
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.router = APIRouter()
        self.connected_clients: List[WebSocket] = []
        self.open_llm_vtuber_main_config = open_llm_vtuber_main_config

        self.preload_models = self.open_llm_vtuber_main_config.get("SERVER", {}).get(
            "PRELOAD_MODELS", False
        )
        if self.preload_models:
            logger.info("Preloading ASR and TTS models...")
            logger.info("Using: " + str(self.open_llm_vtuber_main_config.get("ASR_MODEL")))
            logger.info("Using: " + str(self.open_llm_vtuber_main_config.get("TTS_MODEL")))

            self.model_manager = ModelManager(self.open_llm_vtuber_main_config)
            self.model_manager.initialize_models()

        self._setup_routes()
        self._mount_static_files()
        self.app.include_router(self.router)

    def _setup_routes(self):
        @self.app.websocket("/client-ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "full-text", "text": "Connection established"}))
            self.connected_clients.append(websocket)
            print("Connection established")

            l2d, open_llm_vtuber, _ = self._initialize_components(websocket)
            await websocket.send_text(json.dumps({"type": "set-model", "text": l2d.model_info}))
            print("Model set")

            received_data_buffer = np.array([])
            await websocket.send_text(json.dumps({"type": "control", "text": "start-mic"}))
            conversation_task = None

            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)

                    if data.get("type") == "interrupt-signal":
                        print("Start receiving audio data from front end.")
                        if conversation_task is not None:
                            open_llm_vtuber.interrupt(data.get("text"))

                    elif data.get("type") == "mic-audio-data":
                        received_data_buffer = np.append(
                            received_data_buffer, np.array(list(data.get("audio").values()), dtype=np.float32)
                        )

                    elif data.get("type") == "mic-audio-end" or data.get("type") == "text-input":
                        print("Received audio data end from front end.")
                        await websocket.send_text(json.dumps({"type": "full-text", "text": "Thinking..."}))
                        user_input = data.get("text") if data.get("type") == "text-input" else received_data_buffer
                        received_data_buffer = np.array([])

                        async def _run_conversation():
                            try:
                                await websocket.send_text(json.dumps({"type": "control", "text": "conversation-chain-start"}))
                                await asyncio.to_thread(open_llm_vtuber.conversation_chain, user_input=user_input)
                                await websocket.send_text(json.dumps({"type": "control", "text": "conversation-chain-end"}))
                                print("One Conversation Loop Completed")
                            except asyncio.CancelledError:
                                print("Conversation task was cancelled.")

                        conversation_task = asyncio.create_task(_run_conversation())

            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)

    def _initialize_components(self, websocket: WebSocket):
        l2d = Live2dModel(self.open_llm_vtuber_main_config["LIVE2D_MODEL"])
        custom_asr = self.model_manager.cache.get("asr") if self.preload_models else None
        custom_tts = self.model_manager.cache.get("tts") if self.preload_models else None
        open_llm_vtuber = OpenLLMVTuberMain(self.open_llm_vtuber_main_config, custom_asr=custom_asr, custom_tts=custom_tts)
        audio_preparer = AudioPayloadPreparer()

        return l2d, open_llm_vtuber, audio_preparer

    def _mount_static_files(self):
        self.app.mount("/live2d-models", StaticFiles(directory="live2d-models"), name="live2d-models")
        self.app.mount("/", StaticFiles(directory="./static", html=True), name="static")

    def _start_local_tunnel(self, port):
        try:
            result = subprocess.run(
                ["lt", "--port", str(port)], capture_output=True, text=True, check=True
            )
            url = result.stdout.strip()
            return url
        except Exception as e:
            print(f"Error starting localtunnel: {e}")
            return None

    def run(self, host="127.0.0.1", port=8000):
        logger.info(f"Starting server at {host}:{port}")
        tunnel_url = self._start_local_tunnel(port)
        if tunnel_url:
            print(f"Local tunnel URL: {tunnel_url}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def load_config_with_env(path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    pattern = re.compile(r"\$\{(\w+)\}")

    def replacer(match):
        env_var = match.group(1)
        return os.getenv(env_var, match.group(0))

    content = pattern.sub(replacer, content)
    return yaml.safe_load(content)

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, model: Any) -> None:
        self._cache[key] = model

    def remove(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

class ModelManager:
    def __init__(self, config: Dict):
        self.config = config
        self._old_config = config.copy()
        self.cache = ModelCache()

    def initialize_models(self) -> None:
        if self.config.get("VOICE_INPUT_ON", False):
            self._init_asr()
        if self.config.get("TTS_ON", False):
            self._init_tts()

if __name__ == "__main__":
    atexit.register(lambda: print("Server shutting down..."))

    config = load_config_with_env("conf.yaml")
    config["LIVE2D"] = True  

    server = WebSocketServer(open_llm_vtuber_main_config=config)
    server.run(host=config["HOST"], port=config["PORT"])
                 
