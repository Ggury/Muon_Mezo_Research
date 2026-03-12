import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re



def clean_tensor_string(value):
    """Превращает 'tensor(3.1155, device=...)' в число 3.1155"""
    if isinstance(value, str) and 'tensor' in value:
        # Извлекаем число внутри скобок
        match = re.search(r"tensor\(([\d\.]+)", value)
        if match:
            return float(match.group(1))
    return value

# Устанавливаем стиль графиков
sns.set_theme(style="whitegrid")



def load_and_parse_history(folder_path):
    all_data = []
    files = {
        'AdamW': 'history_adamw.json',
        'Muon': 'history_muon.json',
        'MeZO': 'history_mezo.json',
        'Hybrid': 'history_hybrid.json'
    }

    for label, filename in files.items():
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                df_stats = pd.DataFrame(data['stats'])
                
                # Загружаем loss и очищаем его от строк 'tensor(...)'
                losses = [clean_tensor_string(l) for l in data['loss']]
                df_stats['loss'] = pd.Series(losses)
                
                # На всякий случай очищаем и колонку memory, если там тензоры
                if 'memory' in df_stats.columns:
                    df_stats['memory'] = df_stats['memory'].apply(clean_tensor_string)
                
                df_stats['Optimizer'] = label
                all_data.append(df_stats)
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def plot_results(df):
    # Создаем фигуру с тремя подграфиками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. График Loss (Сходимость)
    # Используем x='step', так как это реальный номер итерации из лога
    sns.lineplot(ax=axes[0], data=df, x='step', y='loss', hue='Optimizer')
    axes[0].set_title('Training Loss Convergence')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss Value')

    # 2. График использования памяти (Peak Memory)
    # В твоем JSON ключ называется 'memory'
    # Группируем по Оптимизатору и находим максимум для каждого
    mem_df = df.groupby('Optimizer')['memory'].max().reset_index()
    
    sns.barplot(ax=axes[1], data=mem_df, x='Optimizer', y='memory', palette='viridis')
    axes[1].set_title('Peak GPU Memory Usage')
    axes[1].set_ylabel('Memory (MB)')
    # Добавим подписи значений над столбиками для точности
    for i, v in enumerate(mem_df['memory']):
        axes[1].text(i, v + 50, f"{int(v)}", ha='center', fontweight='bold')

    # 3. Производительность (Tokens per Second)
    # В твоем JSON есть отличный показатель 'tokens_per_sec'
    if 'tokens_per_sec' in df.columns:
        sns.boxplot(ax=axes[2], data=df, x='Optimizer', y='tokens_per_sec')
        axes[2].set_title('Throughput Comparison')
        axes[2].set_ylabel('Tokens / sec')
    else:
        # Если нет токенов, рисуем время шага (time)
        sns.boxplot(ax=axes[2], data=df, x='Optimizer', y='time')
        axes[2].set_title('Step Latency')
        axes[2].set_ylabel('Time (sec)')

    plt.tight_layout()
    
    # Сохраняем в двух форматах: PNG для быстрого просмотра и PDF для отчета
    plt.savefig('optimization_results.png', dpi=300)
    plt.savefig('optimization_results.pdf')
    print("Графики успешно сохранены в PNG и PDF.")
    plt.show()

# Замени '.' на путь к папке с твоими JSON
df = load_and_parse_history('.')
if df is not None:
    plot_results(df)