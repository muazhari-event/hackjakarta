package app_1

type Response[T any] struct {
	Data T `json:"data"`
}
type Account struct {
	Id       string
	Email    string
	Password string
	WalletId string
}
