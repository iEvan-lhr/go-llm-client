package llm

import (
	"fmt"
	"sync"

	"github.com/ievan-lhr/go-llm-client/providers/dashscope"
	"github.com/ievan-lhr/go-llm-client/providers/generic"
	"github.com/ievan-lhr/go-llm-client/providers/openai"
	"github.com/ievan-lhr/go-llm-client/spec"
)

// clientCache 用于缓存已初始化的客户端，避免重复创建，提高性能。
var (
	clientCache = make(map[string]spec.Client)
	cacheMutex  = &sync.RWMutex{}
)

// GetClient 负责创建和缓存客户端实例。
// 它是导出的，因此 client 包可以使用它。
func GetClient(cfg Config) (spec.Client, error) {
	cacheKey := fmt.Sprintf("%s|%s|%s", cfg.Provider, cfg.APIURL, cfg.APIKey)

	cacheMutex.RLock()
	client, found := clientCache[cacheKey]
	cacheMutex.RUnlock()

	if found {
		return client, nil
	}

	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	client, found = clientCache[cacheKey]
	if found {
		return client, nil
	}

	clientOpts := []spec.ClientOption{
		spec.WithAPIKey(cfg.APIKey),
	}
	if cfg.APIURL != "" {
		clientOpts = append(clientOpts, spec.WithAPIURL(cfg.APIURL))
	}

	var newClient spec.Client
	var err error

	switch cfg.Provider {
	case "dashscope":
		newClient, err = dashscope.NewClient(clientOpts...)
	case "generic":
		newClient, err = generic.NewClient(clientOpts...)
	case "openai":
		newClient, err = openai.NewClient(clientOpts...)
	default:
		return nil, fmt.Errorf("unknown provider: %s", cfg.Provider)
	}

	if err != nil {
		return nil, err
	}

	clientCache[cacheKey] = newClient
	return newClient, nil
}
