Aqui temos uma visualização de quanto mudou a tabela de q learning do agente, ou seja, o quanto o 'conhecimento' que ele tem sobre as melhores ações a serem tomadas mudou.

Para todos os casos, ocorre uma redução progressiva no aprendizado ao longo das épocas, já que o ambiente é novo para o agente.

Conforme as épocas passam, essa mudança vai reduzindo até ficar muito próxima de zero ou ser zero. Este é o ponto de convergência: Quando o conhecimento interno do agente se estabiliza, pois é suficiente para que ele alcance o seu objetivo.

Pode ser observado que quanto maior a taxa de aprendizado, mais rápida é a convergência. o que na visualizaçã abaixo foi extrapolado para uma taxa de aprendizado elevadíssima, como 0.7.

Esse fenômento (quanto maior a taxa de aprendizado mais rápida a convergência) se deve a dois fatores:
1 - O ambiente no qual o agente está inserido é constante. 
2 - O comportamento do agente foi definido como determinístico (epsilon=0)¹.

Se o ambiente fosse dinâmico ou o agente explorasse caminhos não encontrados previamente, qualquer informação estranha ao conhecimento atual do agente (novo obstáculo, barreira, etc.) faria com que conhecimento do agente sobre o ambiente mudasse drasticamente, o que em um problema mais complexo, introduziria grande variabilidade e uma performance volátil.

Quando introduzimos uma taxa de exploraçõ variada, vemos que o agente que aprendeu a solução mais rápido foi o que tinha a menor taxa de exploração (linha amarela, epsilon=0.2).

No entanto, se observarmos este outro plot, observamos um fato interessante: Apesar de os agentes com taxa de exploração mais baixa terem convergido mais rápido (amarelo, azul, verde, marrom), o agente roxo (alpha=0.5 e epsilon=0.8) encontrou uma solução mais eficiente (provavelmente a solução ótima no nosso caso).

Isso exemplifica um dilema clássico no aprendizado por reforço: O equilíbrio entre exploração e exploitação.

Ao mesmo tempo em que o agente pode focar em exploitar o conhecimento já adquirido sobre o ambiente para chegar ao seu objetivo, podem existir planos (ou políticas) melhores, as quais ele ainda não conhece (e sem uma taxa de exploração, nunca irá conhecer).

É muito interessante esse dilema ter se mostrado em um problema simples como este. Em problemas do mundo real, os ambientes são muito mais dinâmicos, complexos, e, por consequência, permitem uma gama de planos e políticas muito mais abrangentes. A decisão entre como equilibrar 


Voltando ao exemplo da figura X, devido à alta taxa de exploração e de aprendizado do agente roxo, ele logo 'esqueceu' a solução que havia encontrado (memória fixa implica que aprendizado rápido leva ao esquecimento rápido, principalmente com uma alta taxa de exploração).

Para evitar que isso aconteça com os agentes, a taxa de exploração é progressivamente diminuída ao longo dos passos (as vezes episódios) de treinamento. Dessa forma, a exploração é priorizada no início, e a exploitação predomina no fim.

