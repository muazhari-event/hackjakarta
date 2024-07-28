# hackjakarta

## Team
- Muhammad Kharisma Azhari (Stackup: muazhari) 
- Vincent Yono (Stackup: vincentyono)

## Idea
### Challenge Statement
> ### Digital Empowerment
> In alignment with Grab’s mission to “Drive Southeast Asia forward by creating economic empowerment for everyone”, your challenge is to harness Generative AI (text, image, or video) to innovate and elevate Grab's Passenger, Driver, or Merchant mobile application’s core journey (e.g., transport, food, mart). ***Develop a creative, scalable solution using Generative AI that improves user experience within the Grab app, drives economic growth, and empowers individuals and businesses within Southeast Asia.***

### Proposed Project
> ### Auto Code Improvement by User Experience and Technical Metrics Optimization
> We propose to improve the user experience and technical metrics of Grab applications. Specifically, we improve user experience and technical metrics using generative AI. Based on our research/literature review, our project hypothetically can contribute to the user experience and economic performance of the company.

### Project Scope
- User Experience Metrics: Error Potential, Latency.
- Technical Metrics: Throughput.

### Project Future Roadmap
- Direct frontend evaluation using reinforcement learning as real-user simulator. Metrics measured by how easy the agent "wants" to be fulfilled.
- Auto system architecture search. Inspired by neural architecture search.

### Tech Stack

- Python
- Golang
- Langchain
- Pymoo
- OpenAI

### Usage
1. Clone the repository
2. Change directory to `./example/app_1` and `./example/app_2`
3. Run `go mod tidy` to install dependencies.
4. Change directory to `./example`
5. Run `pip install -r requirements.txt` to install dependencies.
6. Run cell for `optimization` instantiation in `./example/controller.ipynb`
7. Run `go test ./test` in app_1 and app_2 working directory.
8. Run cell for `optimization.run()` in `./example/controller.ipynb` to start the optimization process.
9. Open dashboard in `http://localhost:{dashboard_port}` to see the optimization process in real-time.
10. Wait until the optimization process is finished.
11. Decide the best solution from the optimization process result.



