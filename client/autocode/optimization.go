package autocode

import (
	"github.com/cosmos72/gomacro/fast"
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

type OptimizationEvaluateResponse struct {
	Objectives            []float64 `json:"objectives"`
	InequalityConstraints []float64 `json:"inequality_onstraints"`
	EqualityConstraints   []float64 `json:"equality_constraints"`
}

type OptimizationApplication interface {
	Duplicate(ctx *OptimizationApplicationContext) any
	Evaluate(ctx *OptimizationApplicationContext) *OptimizationEvaluateResponse
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

func NewOptimization() (optimization *Optimization) {
	optimization = &Optimization{}

	return optimization
}

func (self *Optimization) Prepare() {

}

func (self *Optimization) StartClientServer() {

}

func (self *Optimization) EvaluatePrepare(writer http.ResponseWriter, reader *http.Request) {

}

func (self *Optimization) EvaluateRun(writer http.ResponseWriter, reader *http.Request) {

}

type OptimizationPrepareRequest struct {
	Variables []*OptimizationVariable `json:"variables"`
	Host      string                  `json:"host"`
	Port      int64                   `json:"port"`
}

type OptimizationEvaluateRunRequest struct {
	WorkerId string `json:"worker_id"`
}

type OptimizationEvaluateRunResponse struct {
	Objectives            []float64 `json:"objectives"`
	InequalityConstraints []float64 `json:"inequality_constraints"`
	EqualityConstraints   []float64 `json:"equality_constraints"`
}
