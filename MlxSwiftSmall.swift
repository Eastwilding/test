import Foundation
import TabularData // CSV読み込み
import MLX
import MLXRandom

func tabularDataToMLXArray(_ df: DataFrame,  _ features: [String], _ label: String) -> (features: MLXArray, labels: MLXArray) {
    var featureList = Array<MLXArray>()
    for feature in features {
        //let featureColumn = df[feature].compactMap { $0 as? Double }.map { Float($0) }
        let featureColumn = df[feature].map { Float(($0 as? Double) ?? 0.0) }
        let featureArray = MLXArray(featureColumn)
        let featureArrayReshaped = featureArray.reshaped(-1, 1)
        featureList.append(featureArrayReshaped)
    }
    // axis=1 で結合 (スタック) して [サンプル数, 特徴量数] にする
    let featuresArray = MLX.concatenated(featureList, axis: 1)
        
    // --- ラベル (Labels) の抽出 ---
    let labels = df[label].compactMap { $0 as? Double }.map { Float($0) }
    let labelsArray = MLXArray(labels).reshaped(-1, 1)
        
    return (featuresArray, labelsArray)
}

func loadHousingData() async throws -> DataFrame {
    print("Downloading California Housing CSV data...")
    // 非同期でデータを取得
    let dataURL = URL(string: "https://raw.githubusercontent.com/Eastwilding/test/refs/heads/main/housing.csv")!
    let (data, response) = try await URLSession.shared.data(from: dataURL)
    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
        throw URLError(.badServerResponse)
    }
    print("Download complete. Parsing CSV into DataFrame...")
    // CSVをDataFrameとして読み込み
    let options = CSVReadingOptions(
        hasHeaderRow: true,
        ignoresEmptyLines: true,
        delimiter: ","
    )
    let df = try DataFrame(csvData: data, options: options)
    print("CSV parsing complete. Rows: \(df.rows.count), Columns: \(df.columns.count)")
    return df
}

public func train(_ X: MLXArray, _ y: MLXArray, learningRate: Float = 0.1, epochs: Int = 50){
    let numFeatures = X.shape[1]
    var weights = [String: MLXArray]()
    weights["W"] = randomWeights([numFeatures, 1])
    weights["B"] = MLXArray.zeros([1])
    
    let XScaled = standardize(X)
    let yScaled = standardize(y)
    
    for _ in 1...epochs {
        let (loss, forwardInfo) = forwardLinearRegression(XScaled, yScaled, weights)
        print(loss)
        let gradientsInfo = lossGradients(XScaled, yScaled, weights, forwardInfo)
        for key in weights.keys {
            weights[key] = weights[key]! - learningRate * gradientsInfo[key]!
        }
    }
    print("weights:", weights)
}

public func randomWeights(_ shape: [Int]) -> MLXArray {
    let fanIn = shape[0]
    let fanOut = shape.count > 1 ? shape[1] : 1
    let limit = sqrt(6.0 / Float(fanIn + fanOut))
    
    return MLXRandom.uniform(
        -limit ..< limit,
        shape
    )
}

func standardize(_ X: MLXArray) -> MLXArray {
    let mean = MLX.mean(X, axis: 0, keepDims: true)
    let std  = MLX.std(X, axis: 0, keepDims: true)
    return (X - mean) / (std + MLXArray(1e-8))  // divide-by-zero回避
}

public func lossGradients(
    _ X: MLXArray, _ y: MLXArray, _ weights: [String: MLXArray], _ forwardInfo: [String: MLXArray]) -> [String: MLXArray]{
        let sampleCount = Float(X.shape[0])
        let dLdP = (2 / sampleCount ) * (forwardInfo["P"]! - y)
        //let dLdP = 2 * (forwardInfo["P"]! - y)
        let dPdN = MLXArray.ones(like: forwardInfo["N"]!)
        let dPdB = MLXArray.ones(like: weights["B"]!)
        
        let dLdN = dLdP * dPdN
        let dLdB = (dLdP * dPdB).sum()
        
        let dNdW = X.T
        let dLdW = dNdW.matmul(dLdN)
        
        var gradientsInfo = [String: MLXArray]()
        gradientsInfo["W"] = dLdW
        gradientsInfo["B"] = dLdB
        
        return gradientsInfo
}

public func forwardLinearRegression(
    _ X: MLXArray, _ y: MLXArray, _ weights: [String: MLXArray]) -> (Float, [String: MLXArray]) {
        let N = X.matmul(weights["W"]!)
        let P = N + weights["B"]!
        let loss = (P - y).pow(2).mean()
        
        var forwardInfo = [String: MLXArray]()
        forwardInfo["N"] = N
        forwardInfo["P"] = P
        return (loss.item(Float.self), forwardInfo)
}

public func matrix_function_backward_sum(_ X: MLXArray, _ W: MLXArray) -> MLXArray {
    let N = X.matmul(W)
    let S = sigmoid(N)
    
    let dLdS = MLXArray.ones(like: S)
    let dSdN = S * (1 - S)
    let dLdN = dLdS * dSdN
    
    let dNdX = W.T
    let dLdX = dLdN.matmul(dNdX)
    
    return dLdX
}
