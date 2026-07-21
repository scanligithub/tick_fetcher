package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/injoyai/tdx"
	"github.com/injoyai/tdx/protocol"
)

type StockMaster struct {
	Code     string `json:"code"`
	CodeName string `json:"code_name"`
}

func main() {
	modeFlag := flag.String("mode", "fetch", "运行模式: 'list' (拉取股票清单) 或 'fetch' (拉取历史成交分笔)")
	codesFlag := flag.String("codes", "", "待查询的股票代码列表 (逗号分隔)")
	dateFlag := flag.String("date", "", "查询日期 (格式 YYYYMMDD)")
	outFlag := flag.String("out", "temp_ticks.csv", "导出 CSV 路径")
	serverFlag := flag.String("server", "119.147.212.81:7709", "指定通达信服务器IP:PORT")
	flag.Parse()

	switch *modeFlag {
	case "list":
		runFetchList(*serverFlag)
	case "fetch":
		runFetchTicks(*codesFlag, *dateFlag, *outFlag, *serverFlag)
	default:
		fmt.Println("❌ 未知运行模式")
		os.Exit(1)
	}
}

func runFetchList(server string) {
	cli, err := tdx.Dial(server)
	if err != nil {
		panic(err)
	}
	defer cli.Close()

	var masterList []StockMaster
	exchanges := []protocol.Exchange{protocol.ExchangeSH, protocol.ExchangeSZ, protocol.ExchangeBJ}

	for _, ex := range exchanges {
		resp, err := cli.GetCodeAll(ex)
		if err != nil || resp == nil {
			continue
		}
		for _, item := range resp.List {
			if ex == protocol.ExchangeSH {
				if strings.HasPrefix(item.Code, "60") || strings.HasPrefix(item.Code, "68") {
					masterList = append(masterList, StockMaster{Code: "SH" + item.Code, CodeName: item.Name})
				}
			} else if ex == protocol.ExchangeSZ {
				if strings.HasPrefix(item.Code, "00") || strings.HasPrefix(item.Code, "30") {
					masterList = append(masterList, StockMaster{Code: "SZ" + item.Code, CodeName: item.Name})
				}
			} else if ex == protocol.ExchangeBJ {
				if strings.HasPrefix(item.Code, "43") || strings.HasPrefix(item.Code, "83") ||
					strings.HasPrefix(item.Code, "87") || strings.HasPrefix(item.Code, "88") ||
					strings.HasPrefix(item.Code, "92") {
					masterList = append(masterList, StockMaster{Code: "BJ" + item.Code, CodeName: item.Name})
				}
			}
		}
	}

	file, _ := os.Create("stock_list.json")
	defer file.Close()
	json.NewEncoder(file).Encode(masterList)
}

func runFetchTicks(codesStr, dateStr, outPath, server string) {
	if codesStr == "" || dateStr == "" {
		fmt.Println("❌ 错误: -codes 和 -date 不能为空")
		os.Exit(1)
	}

	rawCodes := strings.Split(codesStr, ",")
	outFile, err := os.Create(outPath)
	if err != nil {
		panic(err)
	}
	defer outFile.Close()

	csvWriter := csv.NewWriter(outFile)
	defer csvWriter.Flush()

	csvWriter.Write([]string{"code", "date", "time", "price", "volume", "status", "number"})

	var mu sync.Mutex
	var wg sync.WaitGroup

	jobChan := make(chan string, len(rawCodes))
	for _, c := range rawCodes {
		jobChan <- strings.TrimSpace(c)
	}
	close(jobChan)

	concurrency := 6
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			workerCli, err := tdx.Dial(server)
			if err != nil {
				return
			}
			defer workerCli.Close()

			for tcode := range jobChan {
				var allTrades protocol.Trades
				start := 0

				for {
					resp, err := workerCli.GetHistoryTrade(dateStr, tcode, uint16(start), 2000)
					if err != nil || resp == nil || len(resp.List) == 0 {
						break
					}
					allTrades = append(allTrades, resp.List...)

					if len(resp.List) < 2000 {
						break
					}
					start += len(resp.List)
					if start > 60000 {
						break
					}
				}

				if len(allTrades) == 0 {
					continue
				}

				var records [][]string
				for _, t := range allTrades {
					timeStr := t.Time.Format("15:04:05")
					priceRaw := fmt.Sprintf("%v", t.Price)
					priceClean := strings.ReplaceAll(priceRaw, "元", "")
					priceClean = strings.TrimSpace(priceClean)

					records = append(records, []string{
						tcode,
						dateStr,
						timeStr,
						priceClean,
						strconv.Itoa(int(t.Volume)),
						strconv.Itoa(t.Status),
						strconv.Itoa(t.Number),
					})
				}

				mu.Lock()
				csvWriter.WriteAll(records)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
}
