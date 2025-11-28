from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from functools import wraps
import traceback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)



MLFLOW_TRACKING_URI = "http://127.0.0.1:5000/"
EXPERIMENT_ID = "467326610704772702"
MODEL_NAME = "diabete_model"

# ================== INICIALIZAÇÃO DO MODELO ==================
class ModelManager:
    """Gerenciador do modelo MLflow com validação e cache."""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        self.feature_names = None
        self.model_loaded_at = None
        self.load_model()
    
    def load_model(self) -> None:
        """Carrega modelo do MLflow com tratamento de erros."""
        try:
            logger.info("🔄 Carregando modelo do MLflow...")
            
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
            
            client = mlflow.client.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME)
            
            if not versions:
                raise ValueError(f"Nenhuma versão encontrada para o modelo '{MODEL_NAME}'")
            
            self.model_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{MODEL_NAME}/{self.model_version}"
            
            self.model = mlflow.sklearn.load_model(model_uri)
            self.feature_names = list(self.model.feature_names_in_)
            self.model_loaded_at = datetime.now()
            
            logger.info(f"Modelo carregado com sucesso: {MODEL_NAME} v{self.model_version}")
            logger.info(f"Features esperadas: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def reload_model(self) -> Dict[str, Any]:
        """Recarrega modelo e retorna informações."""
        logger.info("🔄 Recarregando modelo...")
        old_version = self.model_version
        self.load_model()
        return {
            "previous_version": old_version,
            "current_version": self.model_version,
            "reloaded_at": self.model_loaded_at.isoformat()
        }
    
    def predict(self, df: pd.DataFrame) -> float:
        """Faz predição com validação de features."""
        if self.model is None:
            raise RuntimeError("Modelo não carregado")
        
        # Valida e ordena features
        X = df[self.feature_names]
        pred = self.model.predict_proba(X)[:, 1]
        return float(pred[0])
    
    def is_healthy(self) -> Dict[str, Any]:
        """Verifica saúde do modelo."""
        return {
            "model_loaded": self.model is not None,
            "model_name": MODEL_NAME,
            "model_version": self.model_version,
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "loaded_at": self.model_loaded_at.isoformat() if self.model_loaded_at else None
        }


# ================== VALIDAÇÃO DE ENTRADA ==================
class InputValidator:
    """Validador de entrada de dados."""
    
    @staticmethod
    def validate_prediction_input(data: Dict[str, Any], required_features: list) -> tuple[bool, Optional[str]]:
        """
        Valida entrada de predição.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if not data:
            return False, "Request body está vazio"
        
        if not isinstance(data, dict):
            return False, "Request body deve ser um objeto JSON"
        
        # Verifica features faltantes
        missing_features = set(required_features) - set(data.keys())
        if missing_features:
            return False, f"Features faltantes: {sorted(missing_features)}"
        
        # Verifica features extras
        extra_features = set(data.keys()) - set(required_features)
        if extra_features:
            logger.warning(f"Features extras ignoradas: {sorted(extra_features)}")
        
        # Valida tipos de dados
        for feature in required_features:
            value = data.get(feature)
            if value is None:
                return False, f"Feature '{feature}' não pode ser null"
            
            if not isinstance(value, (int, float)):
                return False, f"Feature '{feature}' deve ser numérica (recebido: {type(value).__name__})"
        
        return True, None


# ================== DECORADORES ==================
def handle_errors(f):
    """Decorator para tratamento centralizado de erros."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Erro de validação: {e}")
            return jsonify({
                "error": "Validation Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
        except RuntimeError as e:
            logger.error(f"Erro de runtime: {e}")
            return jsonify({
                "error": "Runtime Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": "Internal Server Error",
                "message": "Ocorreu um erro inesperado. Verifique os logs.",
                "timestamp": datetime.now().isoformat()
            }), 500
    return decorated_function


def log_request(f):
    """Decorator para logging de requisições."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"{request.method} {request.path} - IP: {request.remote_addr}")
        
        try:
            response = f(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            status = response[1] if isinstance(response, tuple) else 200
            logger.info(f"{request.method} {request.path} - Status: {status} - Tempo: {elapsed:.3f}s")
            
            return response
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"{request.method} {request.path} - Erro após {elapsed:.3f}s: {e}")
            raise
    
    return decorated_function