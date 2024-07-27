package app_gateway

type Response[T any] struct {
	Data T `json:"data"`
}
