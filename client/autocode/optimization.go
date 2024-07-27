package autocode

import (
	"encoding/json"
	"fmt"
	"github.com/cosmos72/gomacro/fast"
	"github.com/gorilla/mux"
	"net/http"
)

const VARIABLE_BINARY = "OptimizationBinary"
const VARIABLE_INTEGER = "OptimizationInteger"
const VARIABLE_REAL = "OptimizationReal"
const VARIABLE_BOOLEAN = "OptimizationBoolean"
const VARIABLE_CHOICE = "OptimizationChoice"
const VALUE_FUNCTION = "function"
const VALUE_BOOLEAN = "boolean"
const VALUE_INTEGER = "integer"
const VALUE_REAL = "real"

type OptimizationVariable struct {
	Id   string `json:"id"`
	Name string `json:"name"`
	Type string `json:"type"`
}

type OptimizationBinary struct {
	OptimizationVariable
}

type OptimizationInteger struct {
	OptimizationVariable
	Bounds [2]int64 `json:"bounds"`
}

type OptimizationReal struct {
	OptimizationVariable
	Bounds [2]float64 `json:"bounds"`
}

type OptimizationChoice struct {
	OptimizationVariable
	Options []*OptimizationValue `json:"options"`
}

type OptimizationValue struct {
	Id   string `json:"id"`
	Type string `json:"type"`
	Data any    `json:"data"`
}

type FunctionValue = func(*OptimizationApplicationContext, ...any) any
type OptimizationFunctionValue struct {
	Function               FunctionValue
	ErrorPotential         float64
	Understandability      float64
	Complexity             float64
	OverallMaintainability float64
	Modularization         float64
}

type OptimizationEvaluateRunResponse struct {
	Objectives            []float64 `json:"objectives"`
	InequalityConstraints []float64 `json:"inequality_onstraints"`
	EqualityConstraints   []float64 `json:"equality_constraints"`
}

type OptimizationApplication interface {
	Duplicate(ctx *OptimizationApplicationContext) any
	Evaluate(ctx *OptimizationApplicationContext) *OptimizationEvaluateRunResponse
}

type OptimizationApplicationContext struct {
	WorkerId               string
	VariableValues         map[string]*OptimizationValue
	ExecutedVariableValues map[string]any
	Optimization           *Optimization
	Application            *OptimizationApplication
}
type Optimization struct {
	ServerHost  string
	ServerPort  int64
	ClientHost  string
	ClientPort  int64
	ServerUrl   string
	ClientUrl   string
	ClientName  string
	Workers     map[string]*OptimizationApplicationContext
	Interpreter *fast.Interp
}

func NewOptimization(
	variables []*OptimizationVariable,
	application OptimizationApplication,
	serverHost string,
	serverPort int64,
	clientHost string,
	clientPort int64,
	clientName string,
) (optimization *Optimization) {
	workers := make(map[string]*OptimizationApplicationContext)
	optimization = &Optimization{
		ServerHost:  serverHost,
		ServerPort:  serverPort,
		ServerUrl:   fmt.Sprintf("%s:%d", serverHost, serverPort),
		ClientHost:  clientHost,
		ClientPort:  clientPort,
		ClientUrl:   fmt.Sprintf("%s:%d", clientHost, clientPort),
		ClientName:  clientName,
		Workers:     workers,
		Interpreter: fast.New(),
	}

	return optimization
}

func (self *Optimization) Prepare() {

}

func (self *Optimization) StartClientServer() {
	router := mux.NewRouter()
	apiRouter := router.PathPrefix("/apis").Subrouter()
	apiRouter.HandleFunc("/optimizations/evaluates/prepares", self.EvaluatePrepare)
	apiRouter.HandleFunc("/optimizations/evaluates/runs", self.EvaluateRun)
	serverErr := http.ListenAndServe(self.ServerUrl, router)
	if serverErr != nil {
		panic(serverErr)
	}
}

func (self *Optimization) EvaluatePrepare(writer http.ResponseWriter, reader *http.Request) {
	responseBody := &OptimizationEvaluatePrepareRequest{}
	decodeErr := json.NewDecoder(reader.Body).Decode(responseBody)
	if decodeErr != nil {
		panic(decodeErr)
	}

	worker := self.Workers[responseBody.WorkerId]
	worker.VariableValues = responseBody.VariableValues
	worker.ExecutedVariableValues = map[string]any{}
}

func (self *Optimization) EvaluateRun(writer http.ResponseWriter, reader *http.Request) {
	responseBody := &OptimizationEvaluateRunRequest{}
	decodeErr := json.NewDecoder(reader.Body).Decode(responseBody)
	if decodeErr != nil {
		panic(decodeErr)
	}

	worker := self.Workers[responseBody.WorkerId]
	response := worker.Application
	encodeErr := json.NewEncoder(writer).Encode(response)
	if encodeErr != nil {
		panic(encodeErr)
	}
}

type OptimizationPrepareRequest struct {
	Variables []*OptimizationVariable `json:"variables"`
	Host      string                  `json:"host"`
	Port      int64                   `json:"port"`
}

type OptimizationEvaluatePrepareRequest struct {
	WorkerId       string                        `json:"worker_id"`
	VariableValues map[string]*OptimizationValue `json:"variafble_values"`
}

type OptimizationEvaluateRunRequest struct {
	WorkerId string `json:"worker_id"`
}
