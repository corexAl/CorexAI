package main

import (
	"encoding/json"
	"net/http"
)

type ChatRequest struct {
	Prompt string `json:"prompt"`
}

type ChatResponse struct {
	Response string `json:"response"`
}

func chatHandler(w http.ResponseWriter, r *http.Request) {

	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}

	var request ChatRequest

	err := json.NewDecoder(r.Body).Decode(&request)

	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Temporary response
	// Later this calls the COREX worker
	response := "COREX: " + request.Prompt

	json.NewEncoder(w).Encode(ChatResponse{
		Response: response,
	})
}

func main() {

	http.HandleFunc("/api/chat", chatHandler)

	println("COREX API listening on :8080")

	http.ListenAndServe(":8080", nil)
}
