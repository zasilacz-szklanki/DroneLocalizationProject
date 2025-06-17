droneData = gTruth.ROILabelData.wall3;

% Przekształć dane ROI na macierz [x, y, width, height]
droneCoords = cell2mat(droneData.Drone);


% Liczba wszystkich klatek
totalFrames = height(droneData);

% Inicjalizuj wektory współrzędnych jako NaN
x_center = NaN(totalFrames, 1);
y_center = NaN(totalFrames, 1);


% Iteruj przez wszystkie klatki
for i = 1:totalFrames
    if ~isempty(droneData.Drone{i})
        % Jeśli Position nie jest puste, oblicz współrzędne środka
        coords = droneData.Drone{i};
        x_center(i) = coords(1) + coords(3) / 2; % x lewego górnego + połowa szerokości
        y_center(i) = coords(2) + coords(4) / 2; % y lewego górnego + połowa wysokości
    end
    % W przeciwnym razie pozostaw NaN (już zainicjalizowane)
end


% Pobierz numery klatek
frameNumbers = (1:totalFrames)';

% Pobierz czas w sekundach od startu
timeSeconds = seconds(droneData.Time - droneData.Time(1));


% Utwórz tabelę
newTable = table(frameNumbers, timeSeconds, x_center, y_center, ...
    'VariableNames', {'frame', 'time', 'x_center', 'y_center'});

writetable(newTable, 'wall3_labels.csv');