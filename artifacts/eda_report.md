
# 📊 Relatório de Análise Exploratória - ED-Insight
## MIMIC-IV-ED Dataset

### 1. Visão Geral do Dataset

| Métrica | Valor |
|---------|-------|
| Total de Visitas ao ED | 425,087 |
| Período | 2110-01-11 01:45:00 a 2212-04-06 14:20:00 |
| Pacientes Únicos | 205,504 |

### 2.  Variáveis Target

#### Lengthened ED Stay (>24h)
- **Casos positivos**: 11,896 (2.80%)
- **Casos negativos**: 413,191 (97.20%)

#### Critical Outcomes
- **Casos positivos**: 165,412 (38.91%)

### 3.  Qualidade dos Dados

| Aspecto | Status |
|---------|--------|
| Missing Values (geral) | Verificar por variável |
| Duplicatas | Mínimas |
| Outliers | Presentes em sinais vitais |

### 4.  Principais Insights

1. **Desbalanceamento**: Classes desbalanceadas requerem técnicas específicas
2. **Padrões Temporais**: Há variação por hora/dia da semana
3. **ESI Score**: Forte preditor de outcomes
4. **Sinais Vitais**: Diferenças significativas entre grupos

### 5.  Próximos Passos

- [ ] Feature Engineering avançado
- [ ] Tratamento de missing values
- [ ] Encoding de variáveis categóricas
- [ ] Seleção de features
- [ ] Treinamento de modelos baseline
