"""
Suite de Testes para API de Predição de Diabetes
Testa todos os endpoints, validações e tratamentos de erro
"""

import os
import sys

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api_request import (  # noqa: E402
    API_BASE_URL,
    TestRunner,
    make_request,
    test_batch_empty_instances,
    test_batch_missing_instances,
    test_batch_partial_errors,
    test_batch_prediction_valid,
    test_empty_body,
    test_endpoint_not_found,
    test_health_check,
    test_invalid_content_type,
    test_invalid_feature_type,
    test_method_not_allowed,
    test_missing_features,
    test_model_info,
    test_null_feature,
    test_response_headers,
    test_valid_prediction,
    valid_prediction_data,
)

# ================== TESTES DE PERFORMANCE ==================


def test_response_time(runner: TestRunner):
    """Testa tempo de resposta."""

    def run():
        import time

        data = valid_prediction_data()
        start = time.time()
        response = make_request("POST", "/predict", json=data)
        elapsed = time.time() - start

        print(f"Status: {response.status_code}")
        print(f"Tempo de resposta: {elapsed:.3f}s")

        assert response.status_code == 200, f"Status esperado 200, recebido {response.status_code}"
        assert elapsed < 2.0, f"Tempo de resposta muito alto: {elapsed:.3f}s (limite: 2s)"

        # Verifica header de tempo
        if "X-Response-Time" in response.headers:
            header_time = response.headers["X-Response-Time"]
            print(f"Header X-Response-Time: {header_time}")

    runner.test("Tempo de Resposta", run)


# ================== MAIN ==================


def run_all_tests():
    """Executa todos os testes."""

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║              SUITE DE TESTES - API DE PREDIÇÃO DE DIABETES                ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)

    runner = TestRunner()

    # Verifica se API está rodando
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=2)
        print(f" API está rodando em {API_BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f" API não está rodando em {API_BASE_URL}")
        print("   Por favor, inicie a API antes de executar os testes:")
        print("   python flask_api.py")
        return

    print("\n🚀 Iniciando testes...\n")

    # Testes básicos
    test_health_check(runner)
    test_model_info(runner)

    # Testes de predição válida
    test_valid_prediction(runner)

    # Testes de validação
    test_missing_features(runner)
    test_invalid_feature_type(runner)
    test_null_feature(runner)
    test_empty_body(runner)
    test_invalid_content_type(runner)

    # Testes de batch
    test_batch_prediction_valid(runner)
    test_batch_missing_instances(runner)
    test_batch_empty_instances(runner)
    test_batch_partial_errors(runner)

    # Testes de erros HTTP
    test_endpoint_not_found(runner)
    test_method_not_allowed(runner)

    # Testes de segurança e performance
    test_response_headers(runner)
    test_response_time(runner)

    # Resumo
    runner.summary()


if __name__ == "__main__":
    run_all_tests()
