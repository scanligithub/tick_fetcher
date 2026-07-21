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
	modeFlag := flag.String("mode", "fetch", "运行模式")
	codesFlag := flag.String("codes", "", "待查询的股票代码列表")
	dateFlag := flag.String("date", "", "查询日期")
	outFlag := flag.String("out", "temp_ticks.csv", "导出 CSV 路径")
	flag.Parse()

	switch *modeFlag {
	case "list":
		runFetchList()
	case "fetch":
		runFetchTicks(*codesFlag, *dateFlag, *outFlag)
	case "kline":
		runFetchKLine(*codesFlag, *outFlag)
	default:
		fmt.Fprintf(os.Stderr, "❌ 未知运行模式\n")
		os.Exit(1)
	}
}

func runFetchList() {
	cli, err := tdx.DialDefault()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: 无法通过 DialDefault 连接: %v\n", err)
		os.Exit(2)
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
			if ex == protocol.ExchangeSH && (strings.HasPrefix(item.Code, "60") || strings.HasPrefix(item.Code, "68")) {
				masterList = append(masterList, StockMaster{Code: "SH" + item.Code, CodeName: item.Name})
			} else if ex == protocol.ExchangeSZ && (strings.HasPrefix(item.Code, "00") || strings.HasPrefix(item.Code, "30")) {
				masterList = append(masterList, StockMaster{Code: "SZ" + item.Code, CodeName: item.Name})
			} else if ex == protocol.ExchangeBJ && (strings.HasPrefix(item.Code, "43") || strings.HasPrefix(item.Code, "83") || strings.HasPrefix(item.Code, "87") || strings.HasPrefix(item.Code, "88") || strings.HasPrefix(item.Code, "92")) {
				masterList = append(masterList, StockMaster{Code: "BJ" + item.Code, CodeName: item.Name})
			}
		}
	}
	file, _ := os.Create("stock_list.json")
	defer file.Close()
	json.NewEncoder(file).Encode(masterList)
}

func runFetchTicks(codesStr, dateStr, outPath string) {
	rawCodes := strings.Split(codesStr, ",")
	outFile, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: 无法创建文件: %v\n", err)
		os.Exit(3)
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
			workerCli, err := tdx.DialDefault()
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

				var records [][]string
				for _, t := range allTrades {
					timeStr := t.Time.Format("15:04:05")
					priceRaw := fmt.Sprintf("%v", t.Price)
					priceClean := strings.TrimSpace(strings.ReplaceAll(priceRaw, "元", ""))
					records = append(records, []string{
						tcode, dateStr, timeStr, priceClean,
						strconv.Itoa(int(t.Volume)), strconv.Itoa(t.Status), strconv.Itoa(t.Number),
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

func runFetchKLine(codesStr, outPath string) {
	rawCodes := strings.Split(codesStr, ",")
	outFile, err := os.Create(outPath)
	if err != nil {
		os.Exit(4)
	}
	defer outFile.Close()

	csvWriter := csv.NewWriter(outFile)
	defer csvWriter.Flush()
	csvWriter.Write([]string{"code", "date", "open", "high", "low", "close", "volume", "amount"})

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
			workerCli, err := tdx.DialDefault()
			if err != nil {
				return
			}
			defer workerCli.Close()

			for tcode := range jobChan {
				market := protocol.ExchangeSH
				if strings.HasPrefix(tcode, "SZ") {
					market = protocol.ExchangeSZ
				} else if strings.HasPrefix(tcode, "BJ") {
					market = protocol.ExchangeBJ
				}
				
				codeNum := tcode
				if len(tcode) > 2 {
					codeNum = tcode[2:]
				}

				resp, err := workerCli.GetKline(uint8(market), codeNum, 0, 100)
				if err != nil || resp == nil {
					continue
				}

				var records [][]string
				for _, k := range resp.List {
					dateStr := k.Time.Format("20060102") 
					// 🚀 黄金终极修复：对 k.Open, k.High, k.Low, k.Close 显式强转为 float64，彻底驱逐 %!f 格式化噪音
					records = append(records, []string{
						tcode, dateStr,
						fmt.Sprintf("%.2f", float64(k.Open)), fmt.Sprintf("%.2f", float64(k.High)),
						fmt.Sprintf("%.2f", float64(k.Low)), fmt.Sprintf("%.2f", float64(k.Close)),
						fmt.Sprintf("%.0f", float64(k.Volume)), 
						fmt.Sprintf("%.2f", float64(k.Amount)), 
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
