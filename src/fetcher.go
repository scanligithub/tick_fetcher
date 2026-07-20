// FILE: src/fetcher.go
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
	flag.Parse()

	switch *modeFlag {
	case "list":
		runFetchList()
	case "fetch":
		runFetchTicks(*codesFlag, *dateFlag, *outFlag)
	default:
		fmt.Println("❌ 未知运行模式。请使用: -mode=list 或 -mode=fetch")
		os.Exit(1)
	}
}

// runFetchList 拉取全市场股票清单并保存为 stock_list.json
func runFetchList() {
	fmt.Println("[Fetcher 内核] 正在获取全市场 active A 股种子清单...")
	cli, err := tdx.DialDefault()
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
	fmt.Printf("[Fetcher 内核] 种子清单解析成功，共计 %d 只标的已落盘。\n", len(masterList))
}

// runFetchTicks 针对指定的股票列表与日期进行分页提取
func runFetchTicks(codesStr, dateStr, outPath string) {
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

	// 写入统一标准的 Tick 数据列头
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
				
				// 分页提取死循环：自动递增偏移量直到当天分笔被 100% 提取完毕
				for {
					resp, err := workerCli.GetHistoryTrade(dateStr, tcode, uint16(start), 2000)
					if err != nil || resp == nil || len(resp.List) == 0 {
						break
					}
					allTrades = append(allTrades, resp.List...)
					
					// 如果抓取的长度小于单页上限 2000 条，说明当天成交已全部拉完
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
					records = append(records, []string{
						tcode,
						dateStr,
						timeStr,
						fmt.Sprintf("%v", t.Price), // 🚀 关键修复：采用默认格式化输出，自适应触发其内部 Stringer 接口，输出纯数字字符串 (例如 1524.22)
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
