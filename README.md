# WPI IQP 2524 — Stock Market Simulation  
**Quantitative Algorithmic Trading Research Project**  
By Brian Wang  

## Abstract  
This project was a five-week stock market simulation centered around the design, implementation, and live deployment of a cloud-based algorithmic trading system, aiming to maximize Sharpe ratio and cumulative net profit. A total of eight strategies were developed iteratively using the QuantConnect platform, employing approaches such as momentum, mean-reversion, statistical arbitrage, and deep reinforcement learning. Four of these strategies were deployed live from June 30th – August 1st, 2025, each for approximately a week. While each had their strengths and were designed to address the limitations of the previously deployed approach, the strategies ultimately struggled to adapt to a market environment defined by resilient bullish momentum, disregard for negative macroeconomic signals, and event-driven volatility. The combined live results — a total return of -5.78% and a Probabilistic Sharpe Ratio of 0% — indicate how even theoretically sound, historically validated algorithms can underperform when market behavior contradicts expectations. This project provided practical experience with the complete algorithmic strategy development process under real market conditions. The perspective and insights gained, along with the undeployed strategies that serve as proof-of-concept models, will act as a foundation for the development of more complex and robust strategies in the future.  

The work culminated in the official WPI report:  
[**Read the full report**](https://digital.wpi.edu/concern/student_works/nz806420r?locale=en)

---

## Technical Stack  
- **Languages & Frameworks:** Python | QuantConnect LEAN | NumPy | pandas | statsmodels | matplotlib  
- **Quantitative Methods:** PCA / IPCA | Hurst Exponent | Ornstein–Uhlenbeck Process | Hidden Markov Models | Reinforcement Learning  
- **Performance Metrics:** Sharpe | Probabilistic Sharpe Ratio (PSR)
- **Deployment:** QuantConnect Cloud | Live Paper Trading via Broker API  

---

## Lessons Learned  
- **Algorithmic Robustness:** Learned to identify weaknesses in algorithm design and refine models under live market feedback rather than rely solely on backtests.  
- **Alpha Generation:** Understood how even theoretically sound, historically validated algorithms can underperform when market behavior contradicts expectations.  
- **Judgment Through Failure:** Developed a strong sense of what *not* to do — recognizing failure patterns, misleading signals, and design pitfalls — leading to stronger intuition and a more disciplined, iterative approach to model development.  
- **Practical Resilience:** Gained firsthand experience debugging live failures, managing deployment errors, and adapting to unexpected technical or market challenges.  
- **Research Foundation:** Built a framework for ongoing experimentation in quantitative trading, setting the stage for future work involving deep learning and regime adaptation.  

---

## Acknowledgements  
Special thanks to Professor Dalin Tang for his continuous guidance, feedback, and support throughout the project, and to Worcester Polytechnic Institute (WPI) for providing the academic infrastructure and resources that made this research possible.  

This project was completed in partial fulfillment of the requirements for the Bachelor of Science degree at Worcester Polytechnic Institute.  
