package rag

import (
"math/rand"
"testing"
"time"
)

// makeBenchPoints generates N random 1024-dim vectors for benchmarking.
func makeBenchPoints(n, dim int) []QdrantPoint {
	pts := make([]QdrantPoint, n)
	for i := range pts {
		v := make([]float32, dim)
		for j := range v {
			v[j] = rand.Float32()
		}
		pts[i] = QdrantPoint{
			ID:      uint64(i),
			Payload: map[string]interface{}{"text": "benchmark"},
			Vector:  v,
			Score:   0,
		}
	}
	return pts
}

// BenchmarkTopNCosine measures topNByCosine on a synthetic corpus of N vectors.
// Reports time per query, plus an implicit cpu% metric via -benchtime scaling.
func BenchmarkTopNCosine(b *testing.B) {
	const dim = 1024
	sizes := []int{10_000, 100_000, 500_000}
	for _, n := range sizes {
		b.Run(fmtSize(n), func(b *testing.B) {
pts := makeBenchPoints(n, dim)
query := make([]float32, dim)
for i := range query {
query[i] = rand.Float32()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := topNByCosine(query, pts, 10, nil)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(n)/b.Elapsed().Seconds()*float64(b.N), "vectors/sec")
		})
	}
}

func fmtSize(n int) string {
	switch {
	case n >= 1_000_000:
		return "1M"
	case n >= 500_000:
		return "500K"
	case n >= 100_000:
		return "100K"
	case n >= 10_000:
		return "10K"
	}
	return time.Duration(n).String()
}
