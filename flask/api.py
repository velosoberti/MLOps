"""
API Flask Profissional para Predição de Diabetes
Com tratamento de erros, validação, logging e health check completo
"""

from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import sys
from functools import wraps
import traceback

# ================== CONFIGURAÇÃO DE LOGGING ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# ================== CONFIGURAÇÕES ==================
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
            
            logger.info(f"✅ Modelo carregado com sucesso: {MODEL_NAME} v{self.model_version}")
            logger.info(f"📋 Features esperadas: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
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
            logger.warning(f"⚠️  Features extras ignoradas: {sorted(extra_features)}")
        
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
            logger.error(f"❌ Erro de validação: {e}")
            return jsonify({
                "error": "Validation Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
        except RuntimeError as e:
            logger.error(f"❌ Erro de runtime: {e}")
            return jsonify({
                "error": "Runtime Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
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
        logger.info(f"📥 {request.method} {request.path} - IP: {request.remote_addr}")
        
        try:
            response = f(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            
            status = response[1] if isinstance(response, tuple) else 200
            logger.info(f"✅ {request.method} {request.path} - Status: {status} - Tempo: {elapsed:.3f}s")
            
            return response
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ {request.method} {request.path} - Erro após {elapsed:.3f}s: {e}")
            raise
    
    return decorated_function


# ================== INICIALIZAÇÃO ==================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Inicializa gerenciador de modelo
try:
    model_manager = ModelManager()
    logger.info("✅ API inicializada com sucesso")
except Exception as e:
    logger.critical(f"💥 Falha crítica ao inicializar API: {e}")
    sys.exit(1)


# ================== ENDPOINTS ==================

@app.route("/health", methods=['GET'])
@log_request
def health_check():
    """
    Health check completo da API.
    
    Verifica:
    - Status da API
    - Conexão com MLflow
    - Status do modelo
    - Disponibilidade de features
    
    Returns:
        JSON com status detalhado
    """
    try:
        # Verifica modelo
        model_health = model_manager.is_healthy()
        
        # Verifica conexão com MLflow
        mlflow_healthy = False
        mlflow_error = None
        try:
            client = mlflow.client.MlflowClient()
            client.get_experiment(EXPERIMENT_ID)
            mlflow_healthy = True
        except Exception as e:
            mlflow_error = str(e)
            logger.error(f"❌ MLflow não acessível: {e}")
        
        # Determina status geral
        all_healthy = model_health["model_loaded"] and mlflow_healthy
        status_code = 200 if all_healthy else 503
        
        response = {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "api": "running",
                "mlflow": {
                    "status": "connected" if mlflow_healthy else "disconnected",
                    "tracking_uri": MLFLOW_TRACKING_URI,
                    "error": mlflow_error
                },
                "model": model_health
            }
        }
        
        if all_healthy:
            logger.info("✅ Health check: HEALTHY")
        else:
            logger.warning("⚠️  Health check: UNHEALTHY")
        
        return jsonify(response), status_code
        
    except Exception as e:
        logger.error(f"❌ Erro no health check: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route("/predict", methods=['POST'])
@log_request
@handle_errors
def predict():
    """
    Endpoint de predição.
    
    Request Body (JSON):
        {
            "Glucose": 148,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Insulin": 0,
            "SkinThickness": 35
        }
    
    Response (JSON):
        {
            "score": 0.6523,
            "prediction": "diabetes",
            "confidence": 0.6523,
            "model_version": 3,
            "timestamp": "2025-11-26T10:30:00"
        }
    """
    # Valida Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Content-Type deve ser 'application/json'",
            "received": request.content_type
        }), 400
    
    # Obtém dados
    data = request.get_json(silent=True)
    
    # Valida entrada
    is_valid, error_message = InputValidator.validate_prediction_input(
        data, 
        model_manager.feature_names
    )
    
    if not is_valid:
        logger.warning(f"⚠️  Validação falhou: {error_message}")
        return jsonify({
            "error": "Invalid Input",
            "message": error_message,
            "expected_features": model_manager.feature_names,
            "received_features": list(data.keys()) if data else []
        }), 400
    
    # Cria DataFrame
    df = pd.DataFrame([data])
    
    # Faz predição
    score = model_manager.predict(df)
    prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"
    
    # Monta resposta
    response = {
        "score": round(score, 4),
        "prediction": prediction_label,
        "confidence": round(score if score >= 0.5 else 1 - score, 4),
        "model_version": model_manager.model_version,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Predição: {prediction_label} (score: {score:.4f})")
    
    return jsonify(response), 200


@app.route("/predict/batch", methods=['POST'])
@log_request
@handle_errors
def predict_batch():
    """
    Endpoint de predição em batch.
    
    Request Body (JSON):
        {
            "instances": [
                {
                    "Glucose": 148,
                    "BMI": 33.6,
                    ...
                },
                {
                    "Glucose": 85,
                    "BMI": 26.6,
                    ...
                }
            ]
        }
    
    Response (JSON):
        {
            "predictions": [
                {
                    "score": 0.6523,
                    "prediction": "diabetes",
                    "instance_index": 0
                },
                ...
            ],
            "total": 2,
            "model_version": 3,
            "timestamp": "2025-11-26T10:30:00"
        }
    """
    # Valida Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Invalid Content-Type",
            "message": "Content-Type deve ser 'application/json'"
        }), 400
    
    data = request.get_json(silent=True)
    
    if not data or 'instances' not in data:
        return jsonify({
            "error": "Invalid Input",
            "message": "Request body deve conter 'instances' (lista de objetos)",
            "example": {
                "instances": [
                    {"Glucose": 148, "BMI": 33.6, "...": "..."}
                ]
            }
        }), 400
    
    instances = data['instances']
    
    if not isinstance(instances, list) or len(instances) == 0:
        return jsonify({
            "error": "Invalid Input",
            "message": "'instances' deve ser uma lista não vazia"
        }), 400
    
    if len(instances) > 1000:
        return jsonify({
            "error": "Batch Too Large",
            "message": f"Máximo de 1000 instâncias por batch (recebido: {len(instances)})"
        }), 400
    
    logger.info(f"📦 Processando batch com {len(instances)} instâncias")
    
    predictions = []
    errors = []
    
    for idx, instance in enumerate(instances):
        try:
            # Valida instância
            is_valid, error_message = InputValidator.validate_prediction_input(
                instance,
                model_manager.feature_names
            )
            
            if not is_valid:
                errors.append({
                    "instance_index": idx,
                    "error": error_message
                })
                continue
            
            # Predição
            df = pd.DataFrame([instance])
            score = model_manager.predict(df)
            prediction_label = "diabetes" if score >= 0.5 else "no_diabetes"
            
            predictions.append({
                "score": round(score, 4),
                "prediction": prediction_label,
                "confidence": round(score if score >= 0.5 else 1 - score, 4),
                "instance_index": idx
            })
            
        except Exception as e:
            logger.error(f"❌ Erro na instância {idx}: {e}")
            errors.append({
                "instance_index": idx,
                "error": str(e)
            })
    
    response = {
        "predictions": predictions,
        "total": len(predictions),
        "errors": errors if errors else None,
        "model_version": model_manager.model_version,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"✅ Batch concluído: {len(predictions)}/{len(instances)} sucessos")
    
    return jsonify(response), 200


@app.route("/model/info", methods=['GET'])
@log_request
def model_info():
    """
    Retorna informações detalhadas do modelo.
    
    Response (JSON):
        {
            "model_name": "log_reg_diabetes_predict",
            "model_version": 3,
            "features": [...],
            "feature_count": 5,
            "loaded_at": "2025-11-26T10:00:00",
            "mlflow_tracking_uri": "http://127.0.0.1:5000/"
        }
    """
    return jsonify({
        "model_name": MODEL_NAME,
        "model_version": model_manager.model_version,
        "features": model_manager.feature_names,
        "feature_count": len(model_manager.feature_names),
        "loaded_at": model_manager.model_loaded_at.isoformat() if model_manager.model_loaded_at else None,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_id": EXPERIMENT_ID
    }), 200


@app.route("/model/reload", methods=['POST'])
@log_request
@handle_errors
def reload_model():
    """
    Recarrega modelo do MLflow (útil após novo treinamento).
    
    Response (JSON):
        {
            "message": "Modelo recarregado com sucesso",
            "previous_version": 3,
            "current_version": 4,
            "reloaded_at": "2025-11-26T10:30:00"
        }
    """
    reload_info = model_manager.reload_model()
    
    return jsonify({
        "message": "Modelo recarregado com sucesso",
        **reload_info
    }), 200


# ================== ERROR HANDLERS ==================

@app.errorhandler(404)
def not_found(e):
    """Handler para rotas não encontradas."""
    logger.warning(f"⚠️  Rota não encontrada: {request.path}")
    return jsonify({
        "error": "Not Found",
        "message": f"Endpoint '{request.path}' não existe",
        "available_endpoints": [
            "GET  /health",
            "GET  /model/info",
            "POST /predict",
            "POST /predict/batch",
            "POST /model/reload"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handler para métodos não permitidos."""
    logger.warning(f"⚠️  Método não permitido: {request.method} {request.path}")
    return jsonify({
        "error": "Method Not Allowed",
        "message": f"Método {request.method} não é permitido para {request.path}",
        "allowed_methods": e.valid_methods if hasattr(e, 'valid_methods') else []
    }), 405


@app.errorhandler(500)
def internal_error(e):
    """Handler para erros internos."""
    logger.error(f"💥 Erro interno: {e}")
    return jsonify({
        "error": "Internal Server Error",
        "message": "Ocorreu um erro interno. Verifique os logs do servidor.",
        "timestamp": datetime.now().isoformat()
    }), 500


# ================== STARTUP ==================

@app.before_request
def before_request():
    """Hook executado antes de cada requisição."""
    request.start_time = datetime.now()


@app.after_request
def after_request(response):
    """Hook executado após cada requisição."""
    if hasattr(request, 'start_time'):
        elapsed = (datetime.now() - request.start_time).total_seconds()
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
    
    # Adiciona headers de segurança
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    
    return response


# ================== MAIN ==================

if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("🚀 Iniciando API de Predição de Diabetes")
    logger.info("=" * 80)
    logger.info(f"📍 Servidor: http://0.0.0.0:5002")
    logger.info(f"🔗 MLflow: {MLFLOW_TRACKING_URI}")
    logger.info(f"📊 Modelo: {MODEL_NAME} v{model_manager.model_version}")
    logger.info(f"📋 Features: {len(model_manager.feature_names)}")
    logger.info("=" * 80)
    logger.info("Endpoints disponíveis:")
    logger.info("  GET  /health")
    logger.info("  GET  /model/info")
    logger.info("  POST /predict")
    logger.info("  POST /predict/batch")
    logger.info("  POST /model/reload")
    logger.info("=" * 80)
    
    app.run(
        host='0.0.0.0',
        port=5005,
        debug=False  # True em desenvolvimento
    )