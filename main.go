// "形状内部饱和度 vs 外围环饱和度"的局部色彩反差算法
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"runtime"
	"sync"
)

// loadImage 从指定路径加载图片（支持 image 包已注册解码器的格式，如 png/jpeg）
func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// saveImage 将图片以 PNG 格式保存到指定路径
func saveImage(img image.Image, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// extractSatBrightMatrices 一次性遍历图片，同时提取饱和度矩阵和亮度矩阵（均为 0-1 浮点值）。
// 相比分别调用两次遍历+HSV转换，这里合并成一次遍历，减少一半的像素读取和重复计算开销；
// 同时去掉了原代码中从未被使用的色相(H)计算，避免无意义的开销。
func extractSatBrightMatrices(img image.Image) (satMat, brightMat [][]float64) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	satMat = make([][]float64, height)
	brightMat = make([][]float64, height)

	for y := range height {
		satMat[y] = make([]float64, width)
		brightMat[y] = make([]float64, width)
		for x := range width {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rf := float64(uint8(r>>8)) / 255
			gf := float64(uint8(g>>8)) / 255
			bf := float64(uint8(b>>8)) / 255

			maxV := math.Max(rf, math.Max(gf, bf))
			minV := math.Min(rf, math.Min(gf, bf))

			v := maxV // 亮度 V = 最大分量
			var s float64
			if maxV != 0 {
				s = (maxV - minV) / maxV // 饱和度 S = (max-min)/max
			}

			satMat[y][x] = s
			brightMat[y][x] = v
		}
	}
	return
}

// findOpaqueBoundingBox 在拼图碎片图（带透明通道）中查找不透明区域的最小外接矩形，
// alphaThreshold 为判定"不透明"的阈值（0-65535，对应 RGBA() 返回的 alpha 范围）。
func findOpaqueBoundingBox(img image.Image, alphaThreshold uint32) image.Rectangle {
	bounds := img.Bounds()
	minX, minY := bounds.Max.X, bounds.Max.Y
	maxX, maxY := bounds.Min.X, bounds.Min.Y
	found := false
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			if a > alphaThreshold {
				found = true
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	if !found {
		return image.Rectangle{}
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// cropImage 按给定矩形裁剪图片，返回一张新的 RGBA 图（坐标从 (0,0) 开始）
func cropImage(img image.Image, rect image.Rectangle) image.Image {
	dst := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	draw.Draw(dst, dst.Bounds(), img, rect.Min, draw.Src)
	return dst
}

// extractShapeMask 从裁剪后的拼图碎片图中提取形状 mask：
// mask[y][x] == true 表示该像素属于拼图形状内部（不透明）
func extractShapeMask(img image.Image, alphaThreshold uint32) [][]bool {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	mask := make([][]bool, height)
	for y := range height {
		mask[y] = make([]bool, width)
		for x := range width {
			_, _, _, a := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			mask[y][x] = a > alphaThreshold
		}
	}
	return mask
}

// dilateMask 对 mask 进行"方形结构元"膨胀（结构元为边长 2*radius+1 的正方形）。
//
// 优化说明：原实现对每个像素都要扫描 (2r+1)×(2r+1) 的邻域，复杂度 O(H·W·r²)。
// 由于方形结构元在数学上是可分离的（separable），膨胀可以拆分为：
//  1. 先做"水平方向"膨胀：每个像素在其左右 radius 范围内只要有一个 true 即为 true；
//  2. 再对第一步结果做"垂直方向"膨胀：每个像素在其上下 radius 范围内只要有一个 true 即为 true。
//
// 两次一维膨胀分别用"行/列前缀和"实现 O(1) 区间查询，整体复杂度降为 O(H·W)，
// 与 radius 大小无关，radius 越大优化效果越明显。
func dilateMask(mask [][]bool, radius int) [][]bool {
	height := len(mask)
	if height == 0 {
		return mask
	}
	width := len(mask[0])

	// 第一步：水平方向膨胀
	horiz := make([][]bool, height)
	for y := range height {
		horiz[y] = make([]bool, width)

		// prefix[i] 表示 mask[y][0..i-1] 中 true 的个数
		prefix := make([]int, width+1)
		for x := range width {
			c := 0
			if mask[y][x] {
				c = 1
			}
			prefix[x+1] = prefix[x] + c
		}

		for x := range width {
			lo := max(x-radius, 0)
			hi := x + radius
			if hi >= width {
				hi = width - 1
			}
			// 区间 [lo, hi] 内是否存在 true，用前缀和差值判断
			horiz[y][x] = prefix[hi+1]-prefix[lo] > 0
		}
	}

	// 第二步：垂直方向膨胀（基于水平膨胀结果按列做同样的前缀和处理）
	result := make([][]bool, height)
	for y := range result {
		result[y] = make([]bool, width)
	}
	for x := range width {
		prefix := make([]int, height+1)
		for y := range height {
			c := 0
			if horiz[y][x] {
				c = 1
			}
			prefix[y+1] = prefix[y] + c
		}
		for y := range height {
			lo := max(y-radius, 0)
			hi := y + radius
			if hi >= height {
				hi = height - 1
			}
			result[y][x] = prefix[hi+1]-prefix[lo] > 0
		}
	}

	return result
}

// MatchResult 表示一次滑动窗口匹配的结果：拼图碎片在背景图中的左上角坐标及匹配得分
type MatchResult struct {
	X, Y  int
	Score float64
}

// offset 用于把二维 mask 展开成坐标偏移列表，避免滑窗过程中反复做二维数组索引和布尔判断
type offset struct {
	dx, dy int
}

// maskToOffsets 将布尔 mask 转换为其取值为 true 的坐标偏移列表
func maskToOffsets(mask [][]bool) []offset {
	var offsets []offset
	for ty := range mask {
		for tx := range mask[ty] {
			if mask[ty][tx] {
				offsets = append(offsets, offset{dx: tx, dy: ty})
			}
		}
	}
	return offsets
}

// matchByLocalContrast 核心算法：滑动窗口，比较"形状内部"与"形状外围环"的饱和度/亮度反差，
// 从而在背景图中定位拼图缺口的位置。
//
// 原理：缺口区域通常经过了"抠图+模糊/变暗"处理，导致缺口内部的饱和度明显低于周围背景，
// 亮度也会有一定差异。因此对每个候选窗口，分别统计：
//   - 形状内部 (shapeMask) 像素的平均饱和度、平均亮度；
//   - 形状外围环 (ringMask，紧贴形状外侧一圈) 像素的平均饱和度、平均亮度；
//
// 若"外围饱和度 - 内部饱和度"越大，且亮度差异也较明显，则该窗口越可能是真正的缺口位置。
//
// bgSat/bgBrightness: 背景图的饱和度矩阵、亮度矩阵
// shapeMask: 拼图形状 mask（true=形状内部）
// ringMask: 外围环形 mask（true=紧邻形状外围的参照区域）
// baseY: 由拼图碎片图推测出的缺口大致 Y 坐标（通常缺口与碎片图在同一行）
// yTolerance: 允许在 baseY 上下浮动搜索的像素范围
// skipLeft: 跳过最左侧一段区域（通常是拼图碎片当前所在的原始位置，避免误匹配到它自己）
//
// 优化说明：
//  1. 将 shapeMask/ringMask 预先展开为坐标偏移列表，滑窗时直接遍历有效点求和，
//     省去了对整张模板逐格做"二维索引 + 双重布尔判断"的开销；
//  2. 由于 x、y 的遍历范围已经保证窗口不会超出背景图边界，内层循环无需再做越界检查；
//  3. 按行区间将搜索任务拆分给多个 goroutine 并行处理，充分利用多核 CPU，
//     最后合并各协程得到的局部最优解，取全局最优。
func matchByLocalContrast(bgSat [][]float64, bgBrightness [][]float64, shapeMask, ringMask [][]bool, baseY, yTolerance, skipLeft int) MatchResult {
	bgHeight := len(bgSat)
	bgWidth := len(bgSat[0])
	tplHeight := len(shapeMask)
	tplWidth := len(shapeMask[0])

	best := MatchResult{X: -1, Y: -1, Score: -math.MaxFloat64}

	yStart := baseY - yTolerance
	yEnd := baseY + yTolerance
	if yStart < 0 {
		yStart = 0
	}
	if yEnd > bgHeight-tplHeight {
		yEnd = bgHeight - tplHeight
	}
	if yStart > yEnd || skipLeft > bgWidth-tplWidth {
		return best
	}

	// 预先展开形状/环形 mask 为坐标偏移列表
	shapeOffsets := maskToOffsets(shapeMask)
	ringOffsets := maskToOffsets(ringMask)
	shapeCount := float64(len(shapeOffsets))
	ringCount := float64(len(ringOffsets))
	if shapeCount == 0 || ringCount == 0 {
		return best
	}

	// 按行区间划分任务，并行搜索
	rowsTotal := yEnd - yStart + 1
	numWorkers := min(runtime.NumCPU(), rowsTotal)
	chunkSize := (rowsTotal + numWorkers - 1) / numWorkers

	var mu sync.Mutex
	var wg sync.WaitGroup

	searchRows := func(yFrom, yTo int) {
		defer wg.Done()
		localBest := MatchResult{X: -1, Y: -1, Score: -math.MaxFloat64}

		for y := yFrom; y <= yTo; y++ {
			for x := skipLeft; x <= bgWidth-tplWidth; x++ {
				var shapeSatSum, shapeBrightSum float64
				for _, o := range shapeOffsets {
					by, bx := y+o.dy, x+o.dx
					shapeSatSum += bgSat[by][bx]
					shapeBrightSum += bgBrightness[by][bx]
				}

				var ringSatSum, ringBrightSum float64
				for _, o := range ringOffsets {
					by, bx := y+o.dy, x+o.dx
					ringSatSum += bgSat[by][bx]
					ringBrightSum += bgBrightness[by][bx]
				}

				shapeAvgSat := shapeSatSum / shapeCount
				ringAvgSat := ringSatSum / ringCount
				shapeAvgBright := shapeBrightSum / shapeCount
				ringAvgBright := ringBrightSum / ringCount

				// 饱和度反差：形状内部饱和度明显低于外围，得分越高
				satDiff := ringAvgSat - shapeAvgSat
				// 亮度反差绝对值：形状内部和外围亮度有差异（不管更亮还是更暗）
				brightDiff := math.Abs(shapeAvgBright - ringAvgBright)

				// 综合得分：主要看饱和度落差，亮度差异作为加分项
				score := satDiff*1.0 + brightDiff*0.3

				if score > localBest.Score {
					localBest.Score = score
					localBest.X = x
					localBest.Y = y
				}
			}
		}

		mu.Lock()
		if localBest.Score > best.Score {
			best = localBest
		}
		mu.Unlock()
	}

	for i := 0; i < numWorkers; i++ {
		from := yStart + i*chunkSize
		if from > yEnd {
			break
		}
		to := from + chunkSize - 1
		if to > yEnd {
			to = yEnd
		}
		wg.Add(1)
		go searchRows(from, to)
	}
	wg.Wait()

	return best
}

// drawRect 在图片副本上沿矩形边框绘制指定颜色的线条，用于标记匹配结果，不修改原图
func drawRect(img image.Image, rect image.Rectangle, lineColor color.Color) *image.RGBA {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Src)
	for x := rect.Min.X; x < rect.Max.X; x++ {
		if x >= bounds.Min.X && x < bounds.Max.X {
			dst.Set(x, rect.Min.Y, lineColor)
			dst.Set(x, rect.Max.Y-1, lineColor)
		}
	}
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		if y >= bounds.Min.Y && y < bounds.Max.Y {
			dst.Set(rect.Min.X, y, lineColor)
			dst.Set(rect.Max.X-1, y, lineColor)
		}
	}
	return dst
}

func main() {
	// 1. 加载背景图（带缺口的完整图）和拼图碎片图（带透明通道）
	bgImg, err := loadImage("bg6.png")
	if err != nil {
		panic(err)
	}
	blockImg, err := loadImage("block6.png")
	if err != nil {
		panic(err)
	}

	fmt.Println("bg尺寸:", bgImg.Bounds())
	fmt.Println("block尺寸:", blockImg.Bounds())

	// 2. 在拼图碎片图中定位不透明区域，得到拼图形状的最小外接矩形
	alphaThreshold := uint32(30000)
	bbox := findOpaqueBoundingBox(blockImg, alphaThreshold)
	fmt.Println("拼图形状边界框:", bbox)
	if bbox.Empty() {
		panic("未找到拼图形状")
	}

	// 3. 裁剪出拼图形状区域，并提取形状 mask
	croppedBlock := cropImage(blockImg, bbox)
	shapeMask := extractShapeMask(croppedBlock, alphaThreshold)

	// 4. 生成外围环形 mask：膨胀后减去原始形状，得到"紧贴形状外侧一圈"的区域，
	//    用作局部对比的参照背景
	dilated := dilateMask(shapeMask, 4) // 膨胀4像素，作为对比参照环
	ringMask := make([][]bool, len(shapeMask))
	for y := range ringMask {
		ringMask[y] = make([]bool, len(shapeMask[0]))
		for x := range ringMask[y] {
			ringMask[y][x] = dilated[y][x] && !shapeMask[y][x]
		}
	}

	// 5. 提取背景图的饱和度矩阵和亮度矩阵（一次遍历同时得到两者）
	bgSat, bgBrightness := extractSatBrightMatrices(bgImg)

	// 拼图碎片在背景图中原本所处的行，与缺口所在行基本一致，作为搜索基准
	baseY := bbox.Min.Y
	// 跳过拼图碎片自身所在的左侧区域，避免把它自己误判为缺口
	skipLeft := bbox.Dx() + 5

	fmt.Println("推测缺口Y坐标:", baseY)

	// 6. 在背景图上滑动窗口搜索，寻找与拼图形状最匹配（饱和度/亮度反差最大）的位置
	result := matchByLocalContrast(bgSat, bgBrightness, shapeMask, ringMask, baseY, 8, skipLeft)
	fmt.Printf("匹配结果 -> X: %d, Y: %d, 得分: %.4f\n", result.X, result.Y, result.Score)

	if result.X < 0 {
		fmt.Println("未找到有效匹配")
		return
	}

	// 7. 在背景图上用矩形框标记出匹配到的缺口位置，方便可视化检查
	matchRect := image.Rect(result.X, result.Y, result.X+bbox.Dx(), result.Y+bbox.Dy())
	marked := drawRect(bgImg, matchRect, color.RGBA{G: 0xff, A: 0xff})
	if err = saveImage(marked, "marked.png"); err != nil {
		fmt.Println("保存标记图失败:", err)
	}

	fmt.Println()
	fmt.Println("======================")
	fmt.Printf("缺口X坐标: %d\n", result.X)
	fmt.Println("======================")
}
