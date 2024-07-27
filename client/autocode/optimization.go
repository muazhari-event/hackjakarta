package autocode

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/cosmos72/gomacro/fast"
	"github.com/google/uuid"
	"github.com/gorilla/mux"
	"net/http"
	"reflect"
)

const VARIABLE_BINARY = "OptimizationBinary"
const VARIABLE_INTEGER = "OptimizationInteger"
const VARIABLE_REAL = "OptimizationReal"
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
	*OptimizationVariable
}

func NewOptimizationBinary(name string) *OptimizationBinary {
	return &OptimizationBinary{
		OptimizationVariable: &OptimizationVariable{
			Id:   uuid.NewString(),
			Name: name,
			Type: VARIABLE_BINARY,
		},
	}
}

type OptimizationInteger struct {
	*OptimizationVariable
	Bounds [2]int64 `json:"bounds"`
}

func NewOptimizationInteger(name string, lowerBound int64, upperBound int64) *OptimizationInteger {
	return &OptimizationInteger{
		OptimizationVariable: &OptimizationVariable{
			Id:   uuid.NewString(),
			Name: name,
			Type: VARIABLE_INTEGER,
		},
		Bounds: [2]int64{lowerBound, upperBound},
	}
}

type OptimizationReal struct {
	*OptimizationVariable
	Bounds [2]float64 `json:"bounds"`
}

func NewOptimizationReal(name string, lowerBound float64, upperBound float64) *OptimizationReal {
	return &OptimizationReal{
		OptimizationVariable: &OptimizationVariable{
			Id:   uuid.NewString(),
			Name: name,
			Type: VARIABLE_REAL,
		},
		Bounds: [2]float64{lowerBound, upperBound},
	}
}

type OptimizationChoice struct {
	*OptimizationVariable
	Options map[string]*OptimizationValue `json:"options"`
}

func getType(value any) string {
	switch value.(type) {
	case OptimizationBinary:
		return VARIABLE_BINARY
	case OptimizationInteger:
		return VARIABLE_INTEGER
	case OptimizationReal:
		return VARIABLE_REAL
	case OptimizationChoice:
		return VARIABLE_CHOICE
	case int64:
		return VALUE_INTEGER
	case float64:
		return VALUE_REAL
	case bool:
		return VALUE_BOOLEAN
	case FunctionValue:
		return VALUE_FUNCTION
	default:
		panic("Unknown type")
	}
}

func NewOptimizationChoice(name string, options []any) *OptimizationChoice {
	transformedOptions := map[string]*OptimizationValue{}
	for _, option := range options {
		optionId := uuid.NewString()
		optionType := getType(option)
		if optionType == VALUE_FUNCTION {
			option = &OptimizationFunctionValue{
				Function:               option.(FunctionValue),
				Complexity:             0,
				ErrorPotential:         0,
				Modularization:         0,
				OverallMaintainability: 0,
				Understandability:      0,
			}
		}
		transformedOptions[optionId] = &OptimizationValue{
			Id:   optionId,
			Type: optionType,
			Data: option,
		}
	}
	return &OptimizationChoice{
		OptimizationVariable: &OptimizationVariable{
			Id:   uuid.NewString(),
			Name: name,
			Type: VARIABLE_CHOICE,
		},
		Options: transformedOptions,
	}
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
	Variables   map[string]any
	Application OptimizationApplication
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
	variables []any,
	application OptimizationApplication,
	serverHost string,
	serverPort int64,
	clientHost string,
	clientPort int64,
	clientName string,
) (optimization *Optimization) {
	workers := make(map[string]*OptimizationApplicationContext)
	transformedVariables := map[string]any{}
	for _, variable := range variables {
		variableId := getFieldValue(variable, "Id").(string)
		transformedVariables[variableId] = variable
	}
	optimization = &Optimization{
		Variables:   transformedVariables,
		Application: application,
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

func getFieldValue(variable any, field string) (output any) {
	reflectedVariable := reflect.ValueOf(variable)
	fieldValue := reflectedVariable.FieldByName(field)
	output = fieldValue.Interface()
	return output
}

func (self *Optimization) Prepare() {
	requestBody := &OptimizationPrepareRequest{
		Variables: self.Variables,
		Host:      self.ClientHost,
		Port:      self.ClientPort,
		Name:      self.ClientName,
	}

	requestBodyMap := requestBody.Map()
	requestBodyJson, jsonErr := json.Marshal(requestBodyMap)
	if jsonErr != nil {
		panic(jsonErr)
	}
	bodyBuffer := bytes.NewBuffer(requestBodyJson)
	client := &http.Client{
		Timeout: 0,
	}
	url := fmt.Sprintf("%s/apis/optimizations/prepares", self.ServerUrl)
	response, responseErr := client.Post(url, "application/json", bodyBuffer)
	if responseErr != nil {
		panic(responseErr)
	}

	if response.StatusCode != 200 {
		panic("Failed to prepare")
	}

	responseBody := &OptimizationPrepareResponse{}
	decodeErr := json.NewDecoder(response.Body).Decode(responseBody)
	if decodeErr != nil {
		panic(decodeErr)
	}

	for _, newVariable := range responseBody.Variables {
		newVariableId := newVariable.(map[string]any)["id"].(string)
		newVariableType := newVariable.(map[string]any)["type"].(string)
		newVariableName := newVariable.(map[string]any)["name"].(string)
		oldVariable, _ := self.Variables[newVariableId]
		if newVariableType == VARIABLE_CHOICE {
			oldOptions := getFieldValue(oldVariable, "Options").(map[string]*OptimizationValue)
			newOptions := map[string]*OptimizationValue{}
			for optionId, option := range newVariable.(map[string]any)["options"].(map[string]any) {
				oldOption, isExists := oldOptions[optionId]
				if isExists == true {
					newOptions[optionId] = oldOption
				} else {
					optionType := option.(map[string]any)["type"].(string)
					var newOptionData any
					if optionType == VALUE_FUNCTION {
						data := option.(map[string]any)["data"].(map[string]any)
						functionName := data["name"].(string)
						functionString := data["string"].(string)
						self.Interpreter.Eval(functionString)
						function, _ := self.Interpreter.Eval1(functionName)
						newOptionData = &OptimizationFunctionValue{
							Function:               function.Interface().(FunctionValue),
							Complexity:             data["complexity"].(float64),
							ErrorPotential:         data["error_potential"].(float64),
							Modularization:         data["modularization"].(float64),
							OverallMaintainability: data["overall_maintainability"].(float64),
							Understandability:      data["understandability"].(float64),
						}
					} else {
						panic(fmt.Sprintf("unsupported option type: %s", optionType))
					}
					newOptions[optionId] = &OptimizationValue{
						Id:   optionId,
						Type: optionType,
						Data: newOptionData,
					}
				}
			}
			newChoice := &OptimizationChoice{
				OptimizationVariable: &OptimizationVariable{
					Id:   newVariableId,
					Type: VARIABLE_CHOICE,
					Name: newVariableName,
				},
				Options: newOptions,
			}
			self.Variables[newVariableId] = newChoice
		}
	}

	for i := int64(0); i < responseBody.NumWorkers; i++ {
		context := &OptimizationApplicationContext{
			WorkerId:               "",
			VariableValues:         nil,
			ExecutedVariableValues: nil,
			Optimization:           nil,
			Application:            nil,
		}
		context.WorkerId = fmt.Sprintf("%d", i)
		context.Optimization = self
		context.Application = self.Application.Duplicate(context).(*OptimizationApplication)
		self.Workers[context.WorkerId] = context
	}

	self.StartClientServer()
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
	Variables map[string]any `json:"variables"`
	Host      string         `json:"host"`
	Port      int64          `json:"port"`
	Name      string         `json:"name"`
}

func (self *OptimizationPrepareRequest) Map() map[string]any {
	transformedVariables := map[string]any{}
	for variableId, variable := range self.Variables {
		variableType := getType(variable)
		switch variableType {
		case VARIABLE_BINARY:
			transformedVariables[variableId] = variable.(*OptimizationBinary)
		case VARIABLE_INTEGER:
			transformedVariables[variableId] = variable.(*OptimizationInteger)
		case VARIABLE_REAL:
			transformedVariables[variableId] = variable.(*OptimizationReal)
		case VARIABLE_CHOICE:
			transformedVariables[variableId] = variable.(*OptimizationChoice)
		default:
			panic("Unknown type")
		}
	}
	return map[string]any{
		"variables": transformedVariables,
		"host":      self.Host,
		"port":      self.Port,
		"name":      self.Name,
	}
}

type OptimizationPrepareResponse struct {
	Variables  map[string]any `json:"variables"`
	NumWorkers int64          `json:"num_workers"`
}

type OptimizationEvaluatePrepareRequest struct {
	WorkerId       string                        `json:"worker_id"`
	VariableValues map[string]*OptimizationValue `json:"variable_values"`
}

type OptimizationEvaluateRunRequest struct {
	WorkerId string `json:"worker_id"`
}
