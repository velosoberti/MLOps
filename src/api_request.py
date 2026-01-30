import json
from typing import Any

import requests

API_BASE_URL = "http://localhost:5005"


class TestRunner:
    """Executor de testes da API."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
        self.tests = []

    def test(self, name: str, func):
        """Executa um teste."""
        print(f"\n{'=' * 80}")
        print(f"{name}")
        print("=" * 80)

        try:
            func()
            self.passed += 1
            print("PASSOU")
            self.tests.append((name, True, None))
        except AssertionError as e:
            self.failed += 1
            print(f"FALHOU: {e}")
            self.tests.append((name, False, str(e)))
        except Exception as e:
            self.failed += 1
            print(f"ERRO: {e}")
            self.tests.append((name, False, str(e)))

    def summary(self):
        """Imprime resumo dos testes."""
        print(f"\n{'=' * 80}")
        print("RESUMO DOS TESTES")
        print("=" * 80)

        for name, passed, error in self.tests:
            status = "✅" if passed else "❌"
            print(f"{status} {name}")
            if error:
                print(f"   └─ {error}")

        print(f"\n{'=' * 80}")
        total = self.passed + self.failed
        print(f"Total: {total} | Passaram: {self.passed} | Falharam: {self.failed}")

        if self.failed == 0:
            print("TODOS OS TESTES PASSARAM!")
        else:
            print(f"⚠️  {self.failed} teste(s) falharam")
        print("=" * 80)


# ================== HELPER FUNCTIONS ==================


def make_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """Faz requisição HTTP."""
    url = f"{API_BASE_URL}{endpoint}"
    return requests.request(method, url, **kwargs)


def valid_prediction_data() -> dict[str, Any]:
    """Retorna dados válidos para predição."""
    return {"Glucose": 148, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Insulin": 0, "SkinThickness": 35}


# ================== TESTES ==================


def test_health_check(runner: TestRunner):
    """Testa endpoint de health check."""

    def run():
        response = make_request("GET", "/health")

        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))

        assert response.status_code in [200, 503], f"Status esperado 200 ou 503, recebido {response.status_code}"
        assert "status" in data, "Resposta deve conter 'status'"
        assert "services" in data, "Resposta deve conter 'services'"
        assert data["status"] in ["healthy", "unhealthy"], f"Status inválido: {data['status']}"

    runner.test("Health Check", run)


def test_model_info(runner: TestRunner):
    """Testa endpoint de informações do modelo."""

    def run():
        response = make_request("GET", "/model/info")

        print(f"Status: {response.status_code}")
        data = response.json()
        print(json.dumps(data, indent=2))

        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "model_name" in data, "Resposta deve conter 'model_name'"
        assert "model_version" in data, "Resposta deve conter 'model_version'"
        assert "features" in data, "Resposta deve conter 'features'"
        assert isinstance(data["features"], list), "Features deve ser uma lista"
        assert len(data["features"]) > 0, "Features não pode estar vazia"

    runner.test("Model Info", run)


def test_valid_prediction(runner: TestRunner):
    """Testa predição com dados válidos."""

    def run():
        data = valid_prediction_data()
        response = make_request("POST", "/predict", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "score" in result, "Resposta deve conter 'score'"
        assert "prediction" in result, "Resposta deve conter 'prediction'"
        assert "confidence" in result, "Resposta deve conter 'confidence'"
        assert "model_version" in result, "Resposta deve conter 'model_version'"

        assert 0 <= result["score"] <= 1, f"Score deve estar entre 0 e 1, recebido {result['score']}"
        assert result["prediction"] in ["diabetes", "no_diabetes"], f"Prediction inválida: {result['prediction']}"

    runner.test("Predição Válida", run)


def test_missing_features(runner: TestRunner):
    """Testa predição com features faltantes."""

    def run():
        data = {"Glucose": 148, "BMI": 33.6}  # Faltam features
        response = make_request("POST", "/predict", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "missing" in result["message"].lower() or "faltantes" in result["message"].lower(), (
            "Mensagem deve indicar features faltantes"
        )

    runner.test("Features Faltantes", run)


def test_invalid_feature_type(runner: TestRunner):
    """Testa predição com tipo de feature inválido."""

    def run():
        data = valid_prediction_data()
        data["DiabetesPedigreeFunction"] = "invalid"  # String ao invés de número
        response = make_request("POST", "/predict", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Tipo de Feature Inválido", run)


def test_null_feature(runner: TestRunner):
    """Testa predição com feature null."""

    def run():
        data = valid_prediction_data()
        data["DiabetesPedigreeFunction"] = None
        response = make_request("POST", "/predict", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "null" in result["message"].lower(), "Mensagem deve indicar valor null"

    runner.test("Feature Null", run)


def test_empty_body(runner: TestRunner):
    """Testa predição com body vazio."""

    def run():
        response = make_request("POST", "/predict", json={})

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Body Vazio", run)


def test_invalid_content_type(runner: TestRunner):
    """Testa predição com Content-Type inválido."""

    def run():
        data = "not a json"
        response = make_request("POST", "/predict", data=data, headers={"Content-Type": "text/plain"})

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Content-Type Inválido", run)


def test_batch_prediction_valid(runner: TestRunner):
    """Testa predição em batch com dados válidos."""

    def run():
        data = {
            "instances": [
                valid_prediction_data(),
                {**valid_prediction_data(), "Glucose": 85},
                {**valid_prediction_data(), "BMI": 26.6},
            ]
        }
        response = make_request("POST", "/predict/batch", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert "predictions" in result, "Resposta deve conter 'predictions'"
        assert "total" in result, "Resposta deve conter 'total'"
        assert len(result["predictions"]) == 3, f"Esperado 3 predições, recebido {len(result['predictions'])}"

        for pred in result["predictions"]:
            assert "score" in pred, "Cada predição deve conter 'score'"
            assert "prediction" in pred, "Cada predição deve conter 'prediction'"
            assert "instance_index" in pred, "Cada predição deve conter 'instance_index'"

    runner.test("Batch Válido", run)


def test_batch_missing_instances(runner: TestRunner):
    """Testa batch sem campo 'instances'."""

    def run():
        response = make_request("POST", "/predict/batch", json={})

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Batch sem Instances", run)


def test_batch_empty_instances(runner: TestRunner):
    """Testa batch com lista vazia."""

    def run():
        response = make_request("POST", "/predict/batch", json={"instances": []})

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 400, f"Status esperado 400, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Batch Vazio", run)


def test_batch_partial_errors(runner: TestRunner):
    """Testa batch com algumas instâncias inválidas."""

    def run():
        data = {
            "instances": [
                valid_prediction_data(),
                {"Glucose": 85},  # Faltam features
                valid_prediction_data(),
            ]
        }
        response = make_request("POST", "/predict/batch", json=data)

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert result["total"] == 2, f"Esperado 2 sucessos, recebido {result['total']}"
        assert result["errors"] is not None, "Deve conter erros"
        assert len(result["errors"]) == 1, f"Esperado 1 erro, recebido {len(result['errors'])}"

    runner.test("Batch com Erros Parciais", run)


def test_endpoint_not_found(runner: TestRunner):
    """Testa endpoint inexistente."""

    def run():
        response = make_request("GET", "/invalid")

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 404, f"Status esperado 404, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"
        assert "available_endpoints" in result, "Deve listar endpoints disponíveis"

    runner.test("Endpoint Inexistente", run)


def test_method_not_allowed(runner: TestRunner):
    """Testa método HTTP não permitido."""

    def run():
        response = make_request("GET", "/predict")  # POST esperado

        print(f"Status: {response.status_code}")
        result = response.json()
        print(json.dumps(result, indent=2))

        assert response.status_code == 405, f"Status esperado 405, recebido {response.status_code}"
        assert "error" in result, "Resposta deve conter 'error'"

    runner.test("Método Não Permitido", run)


def test_response_headers(runner: TestRunner):
    """Testa headers de segurança na resposta."""

    def run():
        response = make_request("GET", "/health")

        print(f"Status: {response.status_code}")
        print("\nHeaders:")
        for key, value in response.headers.items():
            if key.startswith("X-"):
                print(f"  {key}: {value}")

        assert "X-Response-Time" in response.headers, "Deve conter header X-Response-Time"
        assert "X-Content-Type-Options" in response.headers, "Deve conter header X-Content-Type-Options"
        assert "X-Frame-Options" in response.headers, "Deve conter header X-Frame-Options"

    runner.test("Headers de Segurança", run)
