const form = document.getElementById(
    "prediction-form"
);

const resultBox = document.getElementById(
    "result"
);

const historyBody = document.getElementById(
    "history"
);

const refreshButton =
    document.getElementById(
        "refresh-history"
    );


form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        const payload = {
            indoor_temperature:
                Number(
                    document.getElementById(
                        "indoor_temperature"
                    ).value
                ),

            outdoor_temperature:
                Number(
                    document.getElementById(
                        "outdoor_temperature"
                    ).value
                ),

            humidity:
                Number(
                    document.getElementById(
                        "humidity"
                    ).value
                ),

            occupancy:
                Number(
                    document.getElementById(
                        "occupancy"
                    ).value
                ),

            hvac_power_kw:
                Number(
                    document.getElementById(
                        "hvac_power_kw"
                    ).value
                ),
        };


        resultBox.className = "";
        resultBox.textContent =
            "Analyzing sensor data...";


        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body:
                        JSON.stringify(
                            payload
                        ),
                }
            );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    JSON.stringify(
                        data.detail
                    )
                );
            }


            const isAnomaly =
                data.status === "anomaly";


            resultBox.className =
                isAnomaly
                    ? "result-anomaly"
                    : "result-normal";


            resultBox.innerHTML = `
                <strong>Status:</strong>
                ${data.status.toUpperCase()}
                <br>

                <strong>Anomaly Score:</strong>
                ${data.anomaly_score.toFixed(4)}
            `;


            await loadHistory();

        }
        catch (error) {

            resultBox.className =
                "error";

            resultBox.textContent =
                `Error: ${error.message}`;
        }
    }
);


async function loadHistory() {

    try {

        const response =
            await fetch(
                "/predictions"
            );


        if (!response.ok) {

            throw new Error(
                "Unable to load history."
            );
        }


        const predictions =
            await response.json();


        historyBody.innerHTML = "";


        predictions.forEach(
            (prediction) => {

                const row =
                    document.createElement(
                        "tr"
                    );


                const status =
                    prediction.is_anomaly
                        ? "Anomaly"
                        : "Normal";


                const statusClass =
                    prediction.is_anomaly
                        ? "status-anomaly"
                        : "status-normal";


                row.innerHTML = `
                    <td>
                        ${prediction.id}
                    </td>

                    <td>
                        ${prediction.indoor_temperature}
                    </td>

                    <td>
                        ${prediction.outdoor_temperature}
                    </td>

                    <td>
                        ${prediction.humidity}
                    </td>

                    <td>
                        ${prediction.occupancy}
                    </td>

                    <td>
                        ${prediction.hvac_power_kw}
                    </td>

                    <td class="${statusClass}">
                        ${status}
                    </td>

                    <td>
                        ${prediction.anomaly_score.toFixed(4)}
                    </td>
                `;


                historyBody.appendChild(
                    row
                );
            }
        );

    }
    catch (error) {

        historyBody.innerHTML = `
            <tr>
                <td colspan="8">
                    Unable to load prediction history.
                </td>
            </tr>
        `;
    }
}


refreshButton.addEventListener(
    "click",
    loadHistory
);


loadHistory();