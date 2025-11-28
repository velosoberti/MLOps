"""
Script principal para executar predições manuais
Importa o framework e define os dados de exemplo
"""
import sys
from framework.manual_predict_by_api_request import (
    PredictionClient,
    example_single_prediction,
    example_batch_prediction,
    example_custom_prediction,
    example_json_file,
    interactive_menu
)


# ==================== EXEMPLOS PRÉ-DEFINIDOS ====================

EXAMPLE_PATIENTS = {
    "alto_risco": {
        "Glucose": 180,
        "BMI": 38.5,
        "DiabetesPedigreeFunction": 0.85,
        "Insulin": 150,
        "SkinThickness": 40
    },
    "medio_risco": {
        "Glucose": 120,
        "BMI": 30.2,
        "DiabetesPedigreeFunction": 0.45,
        "Insulin": 80,
        "SkinThickness": 28
    },
    "baixo_risco": {
        "Glucose": 85,
        "BMI": 24.5,
        "DiabetesPedigreeFunction": 0.25,
        "Insulin": 40,
        "SkinThickness": 20
    },
    "valores_normais": {
        "Glucose": 95,
        "BMI": 26.0,
        "DiabetesPedigreeFunction": 0.30,
        "Insulin": 50,
        "SkinThickness": 25
    }
}


# ==================== MAIN ====================

def main():
    """Função principal."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║           CLIENTE DE PREDIÇÃO DE DIABETES - USO MANUAL                     ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        # Modo comando
        if sys.argv[1] == "--example":
            example_single_prediction(EXAMPLE_PATIENTS=EXAMPLE_PATIENTS)
            example_batch_prediction(EXAMPLE_PATIENTS=EXAMPLE_PATIENTS)
        elif sys.argv[1] == "--custom":
            example_custom_prediction(EXAMPLE_PATIENTS=EXAMPLE_PATIENTS)
        elif sys.argv[1] == "--file":
            example_json_file(EXAMPLE_PATIENTS=EXAMPLE_PATIENTS)
        elif sys.argv[1] == "--help":
            print("""
USO:
    python manual_predict.py                # Menu interativo
    python manual_predict.py --example      # Rodar exemplos
    python manual_predict.py --custom       # Predição customizada
    python manual_predict.py --file         # Exemplo com arquivo JSON
    python manual_predict.py --help         # Mostrar esta ajuda

EXEMPLOS DE USO PROGRAMÁTICO:
    
    from manual_predict_by_api_request import PredictionClient
    
    # Criar cliente
    client = PredictionClient()
    
    # Predição única
    result = client.predict_single({
        "Glucose": 148,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Insulin": 0,
        "SkinThickness": 35
    })
    
    # Predição em batch
    result = client.predict_batch([
        {...},  # Paciente 1
        {...},  # Paciente 2
    ])

CUSTOMIZAR DADOS DE EXEMPLO:
    
    # Edite o dicionário EXAMPLE_PATIENTS neste arquivo para
    # adicionar seus próprios exemplos de pacientes
    
    EXAMPLE_PATIENTS = {
        "seu_exemplo": {
            "Glucose": 100,
            "BMI": 25.0,
            ...
        }
    }
            """)
        else:
            print(f" Opção inválida: {sys.argv[1]}")
            print("Use --help para ver opções disponíveis")
    else:
        # Modo interativo
        interactive_menu(EXAMPLE_PATIENTS=EXAMPLE_PATIENTS)


if __name__ == "__main__":
    main()