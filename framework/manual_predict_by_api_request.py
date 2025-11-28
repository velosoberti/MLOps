"""
Framework para fazer predições manuais na API de Diabetes
Este módulo contém as classes e funções base para interagir com a API
"""
import requests
import json
from typing import Dict, Any, Optional
import sys

API_BASE_URL = "http://localhost:5005"


class PredictionClient:
    """Cliente para fazer predições na API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self._check_api_health()
    
    def _check_api_health(self):
        """Verifica se a API está funcionando."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"API está rodando em {self.base_url}\n")
            else:
                print(f"API respondeu com status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"Não foi possível conectar à API em {self.base_url}")
            print("   Por favor, inicie a API antes de usar este script:")
            print("   python flask_api.py\n")
            sys.exit(1)
    
    def predict_single(self, data: Dict[str, Any], verbose: bool = True) -> Optional[Dict]:
        """
        Faz uma predição única.
        
        Args:
            data: Dicionário com as features do paciente
            verbose: Se True, imprime informações detalhadas
        
        Returns:
            Dicionário com a resposta da API ou None em caso de erro
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            
            if verbose:
                self._print_single_result(response.status_code, data, result)
            
            return result if response.status_code == 200 else None
            
        except Exception as e:
            if verbose:
                print(f"Erro na requisição: {e}")
            return None
    
    def predict_batch(self, instances: list, verbose: bool = True) -> Optional[Dict]:
        """
        Faz predições em batch.
        
        Args:
            instances: Lista de dicionários com features dos pacientes
            verbose: Se True, imprime informações detalhadas
        
        Returns:
            Dicionário com a resposta da API ou None em caso de erro
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"instances": instances},
                headers={"Content-Type": "application/json"}
            )
            
            result = response.json()
            
            if verbose:
                self._print_batch_result(response.status_code, instances, result)
            
            return result if response.status_code == 200 else None
            
        except Exception as e:
            if verbose:
                print(f"Erro na requisição: {e}")
            return None
    
    def get_model_info(self) -> Optional[Dict]:
        """Obtém informações do modelo."""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Erro ao obter info do modelo: {e}")
            return None
    
    def _print_single_result(self, status_code: int, input_data: Dict, result: Dict):
        """Imprime resultado de predição única."""
        print("="*80)
        print("🔮 RESULTADO DA PREDIÇÃO")
        print("="*80)
        
        print("\n📊 Dados de Entrada:")
        for key, value in input_data.items():
            print(f"   {key}: {value}")
        
        if status_code == 200:
            print(f"\nStatus: {status_code} OK")
            print(f"\nPredição: {result['prediction'].upper()}")
            print(f"core (probabilidade): {result['score']:.4f}")
            print(f"Confiança: {result['confidence']:.2%}")
            print(f"Versão do Modelo: {result['model_version']}")
            
            # Interpretação
            print("\n📋 Interpretação:")
            if result['prediction'] == 'diabetes':
                print(f"    Alta probabilidade de diabetes ({result['score']:.1%})")
                if result['confidence'] > 0.8:
                    print(f"   Confiança ALTA - Recomendado consultar médico")
                elif result['confidence'] > 0.6:
                    print(f"   Confiança MÉDIA - Monitoramento recomendado")
                else:
                    print(f"   Confiança BAIXA - Resultado incerto")
            else:
                print(f"   Baixa probabilidade de diabetes ({result['score']:.1%})")
                if result['confidence'] > 0.8:
                    print(f"   Confiança ALTA - Indicadores normais")
                elif result['confidence'] > 0.6:
                    print(f"   Confiança MÉDIA - Manter hábitos saudáveis")
                else:
                    print(f"   Confiança BAIXA - Monitoramento preventivo")
        else:
            print(f"\nStatus: {status_code} ERRO")
            print(f"\n{result.get('error', 'Erro desconhecido')}")
            if 'message' in result:
                print(f"   Mensagem: {result['message']}")
            if 'details' in result:
                print(f"   Detalhes: {result['details']}")
        
        print("="*80 + "\n")
    
    def _print_batch_result(self, status_code: int, instances: list, result: Dict):
        """Imprime resultado de predição em batch."""
        print("="*80)
        print("🔮 RESULTADO DA PREDIÇÃO EM BATCH")
        print("="*80)
        
        print(f"\nTotal de instâncias enviadas: {len(instances)}")
        
        if status_code == 200:
            print(f"Status: {status_code} OK")
            print(f"\nPredições bem-sucedidas: {result['total']}")
            
            if result.get('errors'):
                print(f"Erros: {len(result['errors'])}")
            
            print("\n" + "─"*80)
            
            # Mostrar cada predição
            for i, pred in enumerate(result['predictions'], 1):
                idx = pred.get('instance_index', i-1)
                instance_data = instances[idx]
                
                print(f"\n🔹 Instância #{i} (índice {idx}):")
                print(f"   Dados: {json.dumps(instance_data, indent=11)[1:-1]}")
                print(f"   Predição: {pred['prediction'].upper()}")
                print(f"   Score: {pred['score']:.4f}")
                print(f"   Confiança: {pred['confidence']:.2%}")
            
            # Mostrar erros se houver
            if result.get('errors'):
                print("\n" + "─"*80)
                print("\nERROS ENCONTRADOS:")
                for error in result['errors']:
                    idx = error['instance_index']
                    print(f"\nInstância #{idx + 1} (índice {idx}):")
                    print(f"   Erro: {error['error']}")
                    if 'details' in error:
                        print(f"   Detalhes: {error['details']}")
            
            # Estatísticas
            print("\n" + "─"*80)
            print("\nESTATÍSTICAS:")
            diabetes_count = sum(1 for p in result['predictions'] if p['prediction'] == 'diabetes')
            no_diabetes_count = result['total'] - diabetes_count
            print(f"   Com diabetes: {diabetes_count} ({diabetes_count/result['total']:.1%})")
            print(f"   Sem diabetes: {no_diabetes_count} ({no_diabetes_count/result['total']:.1%})")
            
            avg_score = sum(p['score'] for p in result['predictions']) / result['total']
            avg_confidence = sum(p['confidence'] for p in result['predictions']) / result['total']
            print(f"   Score médio: {avg_score:.4f}")
            print(f"   Confiança média: {avg_confidence:.2%}")
            
        else:
            print(f"\nStatus: {status_code} ERRO")
            print(f"\n{result.get('error', 'Erro desconhecido')}")
            if 'message' in result:
                print(f"   Mensagem: {result['message']}")
        
        print("\n" + "="*80 + "\n")


# ==================== FUNÇÕES DE DEMONSTRAÇÃO ====================

def example_single_prediction(EXAMPLE_PATIENTS: Dict[str, Dict[str, Any]]):
    """Exemplo de predição única."""
    print("\n" + "="*80)
    print("EXEMPLO 1: PREDIÇÃO ÚNICA")
    print("="*80 + "\n")
    
    client = PredictionClient()
    
    # Predição de um paciente de alto risco
    print("Testando paciente de ALTO RISCO:\n")
    client.predict_single(EXAMPLE_PATIENTS["alto_risco"])
    
    input("\n⏸Pressione ENTER para continuar...")
    
    # Predição de um paciente de baixo risco
    print("\nTestando paciente de BAIXO RISCO:\n")
    client.predict_single(EXAMPLE_PATIENTS["baixo_risco"])


def example_batch_prediction(EXAMPLE_PATIENTS: Dict[str, Dict[str, Any]]):
    """Exemplo de predição em batch."""
    print("\n" + "="*80)
    print("EXEMPLO 2: PREDIÇÃO EM BATCH")
    print("="*80 + "\n")
    
    client = PredictionClient()
    
    instances = [
        EXAMPLE_PATIENTS["alto_risco"],
        EXAMPLE_PATIENTS["medio_risco"],
        EXAMPLE_PATIENTS["baixo_risco"],
        EXAMPLE_PATIENTS["valores_normais"]
    ]
    
    print(f"Testando {len(instances)} pacientes:\n")
    client.predict_batch(instances)


def example_custom_prediction(EXAMPLE_PATIENTS: Dict[str, Dict[str, Any]]):
    """Exemplo de predição customizada."""
    print("\n" + "="*80)
    print("EXEMPLO 3: PREDIÇÃO CUSTOMIZADA")
    print("="*80 + "\n")
    
    client = PredictionClient()
    
    # Obter info do modelo
    model_info = client.get_model_info()
    if model_info:
        print("Features necessárias:")
        for i, feature in enumerate(model_info['features'], 1):
            print(f"   {i}. {feature}")
        print()
    
    print("Digite os valores para cada feature (ou pressione ENTER para usar exemplo):\n")
    
    custom_data = {}
    default_data = EXAMPLE_PATIENTS["medio_risco"]
    
    for feature in ["Glucose", "BMI", "DiabetesPedigreeFunction", "Insulin", "SkinThickness"]:
        value = input(f"   {feature} (padrão: {default_data[feature]}): ").strip()
        
        if value == "":
            custom_data[feature] = default_data[feature]
        else:
            try:
                custom_data[feature] = float(value)
            except ValueError:
                print(f"   Valor inválido, usando padrão: {default_data[feature]}")
                custom_data[feature] = default_data[feature]
    
    print("\nFazendo predição com dados customizados:\n")
    client.predict_single(custom_data)


def example_json_file(EXAMPLE_PATIENTS: Dict[str, Dict[str, Any]]):
    """Exemplo de carregar JSON de arquivo."""
    print("\n" + "="*80)
    print("EXEMPLO 4: CARREGAR DE ARQUIVO JSON")
    print("="*80 + "\n")
    
    # Criar arquivo de exemplo
    example_file = "patient_data.json"
    
    with open(example_file, 'w') as f:
        json.dump(EXAMPLE_PATIENTS["alto_risco"], f, indent=2)
    
    print(f"Arquivo de exemplo criado: {example_file}")
    print(f"\nConteúdo:")
    print(json.dumps(EXAMPLE_PATIENTS["alto_risco"], indent=2))
    
    print(f"\n📖 Lendo dados do arquivo {example_file}...")
    
    with open(example_file, 'r') as f:
        data = json.load(f)
    
    client = PredictionClient()
    print(f"\nFazendo predição com dados do arquivo:\n")
    client.predict_single(data)
    
    print(f"\n Dica: Você pode editar o arquivo {example_file} e rodar novamente!")


def interactive_menu(EXAMPLE_PATIENTS: Dict[str, Dict[str, Any]]):
    """Menu interativo."""
    client = PredictionClient()
    
    while True:
        print("\n" + "="*80)
        print("CLIENTE DE PREDIÇÃO DE DIABETES - MENU INTERATIVO")
        print("="*80)
        print("\nEscolha uma opção:")
        print("   1. Predição única (alto risco)")
        print("   2. Predição única (baixo risco)")
        print("   3. Predição customizada (digite os valores)")
        print("   4. Predição em batch (vários pacientes)")
        print("   5. Carregar JSON de arquivo")
        print("   6. Ver informações do modelo")
        print("   7. Sair")
        print()
        
        choice = input("Opção: ").strip()
        
        if choice == "1":
            print("\nPaciente de ALTO RISCO:\n")
            client.predict_single(EXAMPLE_PATIENTS["alto_risco"])
            
        elif choice == "2":
            print("\nPaciente de BAIXO RISCO:\n")
            client.predict_single(EXAMPLE_PATIENTS["baixo_risco"])
            
        elif choice == "3":
            example_custom_prediction(EXAMPLE_PATIENTS)
            
        elif choice == "4":
            instances = [
                EXAMPLE_PATIENTS["alto_risco"],
                EXAMPLE_PATIENTS["medio_risco"],
                EXAMPLE_PATIENTS["baixo_risco"]
            ]
            client.predict_batch(instances)
            
        elif choice == "5":
            filename = input("\nNome do arquivo JSON: ").strip()
            if filename:
                try:
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    print(f"\nFazendo predição com dados de {filename}:\n")
                    if isinstance(data, list):
                        client.predict_batch(data)
                    else:
                        client.predict_single(data)
                except FileNotFoundError:
                    print(f"Arquivo '{filename}' não encontrado!")
                except json.JSONDecodeError:
                    print(f"Erro ao ler JSON do arquivo '{filename}'!")
            
        elif choice == "6":
            info = client.get_model_info()
            if info:
                print("\n" + "="*80)
                print("INFORMAÇÕES DO MODELO")
                print("="*80)
                print(f"\nModelo: {info['model_name']}")
                print(f"Versão: {info['model_version']}")
                print(f"\nFeatures utilizadas ({len(info['features'])}):")
                for i, feature in enumerate(info['features'], 1):
                    print(f"   {i}. {feature}")
                print("="*80)
            
        elif choice == "7":
            print("\nAté logo!")
            break
            
        else:
            print("\nOpção inválida!")
        
        if choice != "7":
            input("\n⏸Pressione ENTER para continuar...")